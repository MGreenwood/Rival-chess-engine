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
            GameMode::UCI => return Ok(()), // UCI matches don't count in user stats - they're engine vs engine
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
                // Also read UCI games from unified storage
                games.extend(self.read_uci_games_from_unified_storage()?);
            }
            None => {
                // List games from all directories
                games.extend(read_games_from_dir(&self.base_path.join("single_player"))?);
                games.extend(read_games_from_dir(&self.base_path.join("community"))?);
                games.extend(read_games_from_dir(&self.base_path.join("uci_matches"))?);
                // Also include UCI games from unified storage
                games.extend(self.read_uci_games_from_unified_storage()?);
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

    /// Read UCI tournament games from unified storage batches
    fn read_uci_games_from_unified_storage(&self) -> std::io::Result<Vec<GameMetadata>> {
        
        let mut games = Vec::new();
        let unified_dir = self.base_path.join("unified");
        
        if !unified_dir.exists() {
            return Ok(games);
        }

        // Collect batch files first to show progress
        let batch_files: Vec<_> = fs::read_dir(&unified_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.path().file_name()
                    .and_then(|n| n.to_str())
                    .map_or(false, |name| name.starts_with("batch_") && name.ends_with(".json.gz"))
            })
            .collect();

        let total_batches = batch_files.len();
        if total_batches > 0 {
            println!("📦 Reading {} unified batch files for stats cache...", total_batches);
        }

        // Read all batch files with progress
        for (i, entry) in batch_files.into_iter().enumerate() {
            let path = entry.path();
            
            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                println!("📄 Processing batch {}/{}: {}", i + 1, total_batches, filename);
            }
            
            match self.read_unified_batch(&path) {
                Ok(mut batch_games) => {
                    println!("   ✅ Found {} UCI games", batch_games.len());
                    games.append(&mut batch_games);
                }
                Err(e) => {
                    eprintln!("   ⚠️ Warning: Failed to read {}: {}", path.display(), e);
                    // Continue processing other batches
                }
            }
        }

        if !games.is_empty() {
            println!("✅ Loaded {} UCI tournament games from unified storage", games.len());
        }
        
        Ok(games)
    }

    /// Read and parse a single unified storage batch file
    fn read_unified_batch(&self, path: &Path) -> std::io::Result<Vec<GameMetadata>> {
        use flate2::read::GzDecoder;
        use std::io::BufReader;
        
        let file = fs::File::open(path)?;
        let buf_reader = BufReader::new(file);
        let mut decoder = GzDecoder::new(buf_reader);
        let mut contents = String::new();
        std::io::Read::read_to_string(&mut decoder, &mut contents)?;

        // Parse the batch JSON
        let batch: serde_json::Value = serde_json::from_str(&contents)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, 
                format!("Invalid JSON in batch {}: {}", path.display(), e)))?;

        let mut games = Vec::new();

        // Extract games from the batch
        if let Some(batch_games) = batch.get("games").and_then(|g| g.as_array()) {
            for game_value in batch_games {
                if let Ok(metadata) = self.parse_unified_game_to_metadata(game_value) {
                    // Only include UCI tournament games
                    if metadata.mode == GameMode::UCI {
                        games.push(metadata);
                    }
                }
            }
        }

        Ok(games)
    }

    /// Convert unified game format to GameMetadata
    fn parse_unified_game_to_metadata(&self, game_value: &serde_json::Value) -> Result<GameMetadata, Box<dyn std::error::Error>> {
        let game_id = game_value.get("game_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing game_id")?
            .to_string();

        let source = game_value.get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        // Only process UCI tournament games
        if source != "uci_tournament" {
            return Err("Not a UCI tournament game".into());
        }

        let result = game_value.get("result")
            .and_then(|v| v.as_str())
            .unwrap_or("draw");

        let metadata_obj = game_value.get("metadata").ok_or("Missing metadata")?;
        
        let opponent = metadata_obj.get("opponent")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");

        let rival_ai_white = metadata_obj.get("rival_ai_white")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let total_moves = metadata_obj.get("total_moves")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let timestamp_str = game_value.get("timestamp")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Parse timestamp or use current time as fallback
        let timestamp = if !timestamp_str.is_empty() {
            DateTime::parse_from_rfc3339(timestamp_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now())
        } else {
            Utc::now()
        };

        // Convert unified result format to GameStatus
        let status = match result {
            "white_wins" => GameStatus::WhiteWins,
            "black_wins" => GameStatus::BlackWins,
            _ => GameStatus::DrawStalemate,
        };

        // Determine player color and name
        let player_color = if rival_ai_white { "white" } else { "black" };
        let player_name = Some(format!("RivalAI vs {}", opponent));

        Ok(GameMetadata {
            game_id,
            mode: GameMode::UCI,
            created_at: timestamp,
            last_move_at: timestamp,
            status,
            total_moves,
            player_color: player_color.to_string(),
            player_name,
            engine_version: "1.0.0".to_string(),
        })
    }
} 