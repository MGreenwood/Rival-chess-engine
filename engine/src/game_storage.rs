use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum GameMode {
    SinglePlayer,
    Community,
    UCI,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum GameStatus {
    Active,
    Waiting,
    WhiteWins,
    BlackWins,
    DrawStalemate,
    DrawInsufficientMaterial,
    DrawRepetition,
    DrawFiftyMoves,
}

impl From<&str> for GameStatus {
    fn from(s: &str) -> Self {
        match s {
            "active" => GameStatus::Active,
            "waiting" => GameStatus::Waiting,
            "white_wins" => GameStatus::WhiteWins,
            "black_wins" => GameStatus::BlackWins,
            "draw_stalemate" => GameStatus::DrawStalemate,
            "draw_insufficient" => GameStatus::DrawInsufficientMaterial,
            "draw_repetition" => GameStatus::DrawRepetition,
            "draw_fifty_moves" => GameStatus::DrawFiftyMoves,
            _ => GameStatus::Active,
        }
    }
}

impl ToString for GameStatus {
    fn to_string(&self) -> String {
        match self {
            GameStatus::Active => "active",
            GameStatus::Waiting => "waiting",
            GameStatus::WhiteWins => "white_wins",
            GameStatus::BlackWins => "black_wins",
            GameStatus::DrawStalemate => "draw_stalemate",
            GameStatus::DrawInsufficientMaterial => "draw_insufficient",
            GameStatus::DrawRepetition => "draw_repetition",
            GameStatus::DrawFiftyMoves => "draw_fifty_moves",
        }.to_string()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GameMetadata {
    pub game_id: String,
    pub mode: GameMode,
    pub created_at: DateTime<Utc>,
    pub last_move_at: DateTime<Utc>,
    pub status: GameStatus,
    pub total_moves: usize,
    pub player_color: String,
    pub player_name: Option<String>,
    pub engine_version: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GameState {
    pub metadata: GameMetadata,
    pub board: String,
    pub move_history: Vec<String>,
    pub analysis: Option<HashMap<String, f32>>, // Position evaluation history
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CommunityGameState {
    pub metadata: GameMetadata,
    pub board: String,
    pub move_history: Vec<String>,
    pub votes: HashMap<String, Vec<String>>, // move -> voter_ids
    pub voting_ends_at: Option<DateTime<Utc>>,
}

// Persistent model statistics that survive game archival
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct ModelStats {
    pub total_games: usize,
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub games_by_engine_version: HashMap<String, EngineVersionStats>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct EngineVersionStats {
    pub total_games: usize,
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub first_game: DateTime<Utc>,
    pub last_game: DateTime<Utc>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct PersistentStats {
    pub single_player: ModelStats,
    pub community: ModelStats,
}

pub struct GameStorage {
    base_path: PathBuf,
    stats_path: PathBuf,
}

impl GameStorage {
    pub fn new<P: AsRef<Path>>(base_path: P) -> std::io::Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        fs::create_dir_all(&base_path)?;
        let stats_path = base_path.join("model_stats.json");
        Ok(Self { base_path, stats_path })
    }

    fn get_game_path(&self, game_id: &str, mode: &GameMode) -> PathBuf {
        let mode_dir = match mode {
            GameMode::SinglePlayer => "single_player",
            GameMode::Community => "community",
            GameMode::UCI => "uci_matches",
        };
        self.base_path.join(mode_dir).join(format!("{}.json", game_id))
    }

    pub fn save_game(&self, state: &GameState) -> std::io::Result<()> {
        let path = self.get_game_path(&state.metadata.game_id, &state.metadata.mode);
        fs::create_dir_all(path.parent().unwrap())?;
        let json = serde_json::to_string_pretty(state)?;
        fs::write(path, json)?;
        
        // Update persistent stats when saving completed games
        if matches!(state.metadata.status, 
            GameStatus::WhiteWins | GameStatus::BlackWins | 
            GameStatus::DrawStalemate | GameStatus::DrawInsufficientMaterial |
            GameStatus::DrawRepetition | GameStatus::DrawFiftyMoves) {
            self.update_persistent_stats(&state.metadata)?;
        }
        
        Ok(())
    }

    pub fn save_community_game(&self, state: &CommunityGameState) -> std::io::Result<()> {
        let path = self.get_game_path(&state.metadata.game_id, &state.metadata.mode);
        fs::create_dir_all(path.parent().unwrap())?;
        let json = serde_json::to_string_pretty(state)?;
        fs::write(path, json)?;
        
        // Update persistent stats when saving completed games
        if matches!(state.metadata.status, 
            GameStatus::WhiteWins | GameStatus::BlackWins | 
            GameStatus::DrawStalemate | GameStatus::DrawInsufficientMaterial |
            GameStatus::DrawRepetition | GameStatus::DrawFiftyMoves) {
            self.update_persistent_stats(&state.metadata)?;
        }
        
        Ok(())
    }

    fn update_persistent_stats(&self, metadata: &GameMetadata) -> std::io::Result<()> {
        let mut stats = self.load_persistent_stats()?;
        
        let model_stats = match metadata.mode {
            GameMode::SinglePlayer => &mut stats.single_player,
            GameMode::Community => &mut stats.community,
            GameMode::UCI => &mut stats.single_player, // UCI matches count as single player for stats
        };
        
        // Update overall stats
        model_stats.total_games += 1;
        match metadata.status {
            GameStatus::WhiteWins => {
                if metadata.player_color == "white" {
                    model_stats.wins += 1;
                } else {
                    model_stats.losses += 1;
                }
            },
            GameStatus::BlackWins => {
                if metadata.player_color == "black" {
                    model_stats.wins += 1;
                } else {
                    model_stats.losses += 1;
                }
            },
            _ => model_stats.draws += 1,
        }
        
        // Update engine version specific stats
        let engine_stats = model_stats.games_by_engine_version
            .entry(metadata.engine_version.clone())
            .or_insert_with(|| EngineVersionStats {
                first_game: metadata.created_at,
                last_game: metadata.created_at,
                ..Default::default()
            });
            
        engine_stats.total_games += 1;
        engine_stats.last_game = metadata.last_move_at;
        
        match metadata.status {
            GameStatus::WhiteWins => {
                if metadata.player_color == "white" {
                    engine_stats.wins += 1;
                } else {
                    engine_stats.losses += 1;
                }
            },
            GameStatus::BlackWins => {
                if metadata.player_color == "black" {
                    engine_stats.wins += 1;
                } else {
                    engine_stats.losses += 1;
                }
            },
            _ => engine_stats.draws += 1,
        }
        
        model_stats.last_updated = Utc::now();
        
        // Save updated stats
        self.save_persistent_stats(&stats)
    }

    pub fn load_persistent_stats(&self) -> std::io::Result<PersistentStats> {
        if self.stats_path.exists() {
            let json = fs::read_to_string(&self.stats_path)?;
            Ok(serde_json::from_str(&json).unwrap_or_default())
        } else {
            Ok(PersistentStats::default())
        }
    }

    pub fn save_persistent_stats(&self, stats: &PersistentStats) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(stats)?;
        fs::write(&self.stats_path, json)
    }

    pub fn archive_games_metadata(&self, games: &[GameMetadata]) -> std::io::Result<()> {
        // Ensure all game metadata is recorded in persistent stats before archival
        for metadata in games {
            if matches!(metadata.status, 
                GameStatus::WhiteWins | GameStatus::BlackWins | 
                GameStatus::DrawStalemate | GameStatus::DrawInsufficientMaterial |
                GameStatus::DrawRepetition | GameStatus::DrawFiftyMoves) {
                self.update_persistent_stats(metadata)?;
            }
        }
        Ok(())
    }

    pub fn load_game(&self, game_id: &str, mode: &GameMode) -> std::io::Result<GameState> {
        let path = self.get_game_path(game_id, mode);
        let json = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }

    pub fn load_community_game(&self, game_id: &str) -> std::io::Result<CommunityGameState> {
        let path = self.get_game_path(game_id, &GameMode::Community);
        let json = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }

    pub fn list_games(&self, mode: Option<GameMode>) -> std::io::Result<Vec<GameMetadata>> {
        let mut games = Vec::new();

        // Helper function to read games from a directory
        let read_games_from_dir = |dir_path: &Path| -> std::io::Result<Vec<GameMetadata>> {
            let mut dir_games = Vec::new();
            if dir_path.exists() {
                for entry in fs::read_dir(dir_path)? {
                    let entry = entry?;
                    if entry.path().extension().map_or(false, |ext| ext == "json") {
                        let json = fs::read_to_string(entry.path())?;
                        if let Ok(game) = serde_json::from_str::<GameState>(&json) {
                            dir_games.push(game.metadata);
                        }
                    }
                }
            }
            Ok(dir_games)
        };

        match mode {
            Some(GameMode::SinglePlayer) => {
                games.extend(read_games_from_dir(&self.base_path.join("single_player"))?);
            }
            Some(GameMode::Community) => {
                games.extend(read_games_from_dir(&self.base_path.join("community"))?);
            }
            Some(GameMode::UCI) => {
                games.extend(read_games_from_dir(&self.base_path.join("uci_matches"))?);
            }
            None => {
                // List games from all directories
                games.extend(read_games_from_dir(&self.base_path.join("single_player"))?);
                games.extend(read_games_from_dir(&self.base_path.join("community"))?);
                games.extend(read_games_from_dir(&self.base_path.join("uci_matches"))?);
            }
        }

        // Sort by last move, most recent first
        games.sort_by(|a, b| b.last_move_at.cmp(&a.last_move_at));
        Ok(games)
    }

    pub fn delete_game(&self, game_id: &str, mode: &GameMode) -> std::io::Result<()> {
        let path = self.get_game_path(game_id, mode);
        if path.exists() {
            fs::remove_file(path)?;
        }
        Ok(())
    }

    pub fn generate_game_id() -> String {
        Uuid::new_v4().to_string()
    }
} 