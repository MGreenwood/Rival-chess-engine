#!/usr/bin/env python3
"""
Test script to verify aggressive self-play generation works correctly.
"""

import requests
import time
import json

def test_aggressive_selfplay():
    """Test that self-play generation starts aggressively when conditions are met."""
    print("🧪 Testing aggressive self-play generation...")
    
    # Wait a bit for server to initialize
    print("⏳ Waiting 5 seconds for server to initialize...")
    time.sleep(5)
    
    for i in range(10):  # Test for 100 seconds (10 checks x 10 seconds)
        try:
            print(f"\n📊 Check {i+1}/10:")
            
            # Get self-play status
            response = requests.get("http://localhost:3000/self-play-status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                print(f"   Current games: {data['current_self_play_games']}")
                print(f"   Active players: {data['active_players']}")
                print(f"   Is training: {data['is_training']}")
                print(f"   Is generating: {data['is_generating']}")
                print(f"   Community busy: {data['community_busy']}")
                print(f"   Can generate: {data['can_generate']}")
                print(f"   Status: {data['status']}")
                
                # Check if generation should be happening
                if not data['is_training'] and not data['community_busy'] and data['current_self_play_games'] > 0:
                    if data['is_generating']:
                        print("   ✅ GENERATION IN PROGRESS!")
                    elif data['can_generate']:
                        print("   ⏳ Should start generating soon...")
                    else:
                        print("   ❓ Conditions met but not generating?")
                else:
                    print("   ⏸️ Waiting for conditions to be met")
                
            else:
                print(f"   ❌ Failed to get status: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        if i < 9:  # Don't sleep after the last check
            print("   ⏰ Waiting 10 seconds for next check...")
            time.sleep(10)
    
    print("\n🏁 Test completed. Check server logs for detailed generation activity.")

if __name__ == "__main__":
    test_aggressive_selfplay() 