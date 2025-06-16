pub mod python;

// Remove unused re-export
// pub use python::ModelBridge;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bridge_creation() {
        let bridge = ModelBridge::new("models/checkpoint.pt").unwrap();
        assert_eq!(bridge.device, "cpu");
    }
} 