#!/usr/bin/env python3
"""
Debug script to check self-play generation status
"""

import requests
import time
import json

def check_selfplay_status():
    """Check the self-play status from the server"""
    try:
        response = requests.get("http://localhost:3000/self-play-status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("📊 Self-Play Status:")
            print(f"   Current games: {data['current_self_play_games']}")
            print(f"   Active players: {data['active_players']}")
            print(f"   GPU utilization: {data['gpu_utilization']*100:.1f}%")
            print(f"   Is training: {data['is_training']}")
            print(f"   Status: {data['status']}")
            return data
        else:
            print(f"❌ Failed to get status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return None

def check_community_game():
    """Check if community game is blocking self-play"""
    try:
        response = requests.get("http://localhost:3000/api/community/state", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("🏛️ Community Game Status:")
            print(f"   Engine thinking: {data['engine_thinking']}")
            print(f"   Can vote: {data['can_vote']}")
            print(f"   Voting phase: {data['is_voting_phase']}")
            print(f"   Status: {data['status']}")
            return data['engine_thinking']
        else:
            print(f"❌ Failed to get community status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking community: {e}")
        return False

def main():
    print("🔍 Debugging Self-Play Generation...")
    print("="*50)
    
    for i in range(5):  # Check 5 times over 5 minutes
        print(f"\n📋 Check #{i+1} at {time.strftime('%H:%M:%S')}")
        
        # Check self-play status
        selfplay_data = check_selfplay_status()
        
        # Check community game
        community_busy = check_community_game()
        
        # Analysis
        if selfplay_data:
            current_games = selfplay_data['current_self_play_games']
            active_players = selfplay_data['active_players']
            is_training = selfplay_data['is_training']
            
            print("\n🔬 Analysis:")
            if current_games == 0:
                print("   ⚠️ Zero self-play games configured!")
            elif community_busy:
                print("   🛡️ Self-play blocked by community engine")
            elif is_training:
                print("   🎓 Self-play reduced due to training")
            elif active_players > 0:
                print(f"   👥 {active_players} players active - scaled mode")
            else:
                print("   ✅ Conditions look good for self-play")
                
            print(f"   Expected behavior: {current_games} games every 60s")
        
        if i < 4:  # Don't wait after last check
            print("   ⏳ Waiting 60 seconds...")
            time.sleep(60)
    
    print("\n🎯 Recommendations:")
    print("1. Check server logs for Python script errors")
    print("2. Verify --enable-self-play flag is set correctly")
    print("3. Check if training_games directory exists and is writable")
    print("4. Monitor for 'generate_self_play_games' calls in logs")

if __name__ == "__main__":
    main() 