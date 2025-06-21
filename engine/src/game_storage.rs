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

pub struct GameStorage {
    base_path: PathBuf,
}

impl GameStorage {
    pub fn new<P: AsRef<Path>>(base_path: P) -> std::io::Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        fs::create_dir_all(&base_path)?;
        Ok(Self { base_path })
    }

    fn get_game_path(&self, game_id: &str, mode: &GameMode) -> PathBuf {
        let mode_dir = match mode {
            GameMode::SinglePlayer => "single_player",
            GameMode::Community => "community",
        };
        self.base_path.join(mode_dir).join(format!("{}.json", game_id))
    }

    pub fn save_game(&self, state: &GameState) -> std::io::Result<()> {
        let path = self.get_game_path(&state.metadata.game_id, &state.metadata.mode);
        fs::create_dir_all(path.parent().unwrap())?;
        let json = serde_json::to_string_pretty(state)?;
        fs::write(path, json)
    }

    pub fn save_community_game(&self, state: &CommunityGameState) -> std::io::Result<()> {
        let path = self.get_game_path(&state.metadata.game_id, &state.metadata.mode);
        fs::create_dir_all(path.parent().unwrap())?;
        let json = serde_json::to_string_pretty(state)?;
        fs::write(path, json)
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
            None => {
                // List games from both directories
                games.extend(read_games_from_dir(&self.base_path.join("single_player"))?);
                games.extend(read_games_from_dir(&self.base_path.join("community"))?);
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