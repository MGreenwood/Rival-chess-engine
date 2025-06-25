#!/usr/bin/env python3
"""
PAG Performance Benchmark
Test how fast we can build ultra-dense PAGs with the new sophisticated edge relationship system
"""

import time
import statistics
from python.src.rival_ai.pag import PositionalAdjacencyGraph, PAGConfig
import chess

def benchmark_pag_construction():
    """Benchmark PAG construction performance across different position types"""
    
    # Test positions of varying complexity
    test_positions = [
        # Starting position
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        
        # Complex middlegame
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4",
        
        # Tactical position with pins/forks
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 3 4",
        
        # Endgame position
        "8/2k5/3p4/p2P1p2/P3pP2/8/2K5/8 w - - 0 1",
        
        # Complex position with many pieces
        "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP1QPPP/R1B1K2R w KQ - 0 8",
    ]
    
    position_names = [
        "Starting Position",
        "Complex Middlegame", 
        "Tactical Position",
        "Endgame",
        "Many Pieces"
    ]
    
    print("🚀 PAG Construction Performance Benchmark")
    print("=" * 60)
    
    # Create PAG config for enhanced features
    pag_config = PAGConfig()
    
    overall_times = []
    
    for i, (fen, name) in enumerate(zip(test_positions, position_names)):
        print(f"\n📋 Testing: {name}")
        print(f"FEN: {fen}")
        
        # Warm up
        board = chess.Board(fen)
        pag = PositionalAdjacencyGraph(board, pag_config)
        
        # Benchmark multiple runs
        times = []
        num_runs = 10
        
        for run in range(num_runs):
            board = chess.Board(fen)
            start_time = time.perf_counter()
            
            pag = PositionalAdjacencyGraph(board, pag_config)
            
            end_time = time.perf_counter()
            construction_time = (end_time - start_time) * 1000  # Convert to milliseconds
            times.append(construction_time)
        
        # Calculate statistics
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        
        overall_times.extend(times)
        
        # Get PAG statistics
        num_pieces = len(pag.piece_nodes)
        num_critical_squares = len(pag.critical_square_nodes)
        
        # Count total edges across all edge types
        num_edges = (len(pag.direct_edges) + len(pag.control_edges) + 
                    len(pag.mobility_edges) + len(pag.cooperative_edges) + 
                    len(pag.obstructive_edges) + len(pag.vulnerability_edges) + 
                    len(pag.pawn_structure_edges) + len(pag.king_safety_edges) + 
                    len(pag.center_control_edges) + len(pag.material_tension_edges))
        
        print(f"⏱️  Average Time: {avg_time:.2f}ms")
        print(f"📊 Range: {min_time:.2f}ms - {max_time:.2f}ms (±{std_dev:.2f}ms)")
        print(f"🎯 PAG Stats: {num_pieces} pieces, {num_critical_squares} critical squares, {num_edges} edges")
        print(f"📈 Edges per piece: {num_edges/max(num_pieces,1):.1f}")
        
        # Edge type breakdown
        edge_breakdown = {
            'Direct': len(pag.direct_edges),
            'Control': len(pag.control_edges),
            'Mobility': len(pag.mobility_edges),
            'Cooperation': len(pag.cooperative_edges),
            'Obstruction': len(pag.obstructive_edges),
            'Vulnerability': len(pag.vulnerability_edges),
            'Pawn Structure': len(pag.pawn_structure_edges),
            'King Safety': len(pag.king_safety_edges),
            'Center Control': len(pag.center_control_edges),
            'Material Tension': len(pag.material_tension_edges),
        }
        
        print(f"🔗 Edge Types: {', '.join(f'{k}:{v}' for k,v in edge_breakdown.items() if v > 0)}")
        
        # Performance rating
        if avg_time < 10:
            rating = "🚀 EXCELLENT"
        elif avg_time < 50:
            rating = "✅ GOOD"
        elif avg_time < 100:
            rating = "⚠️  ACCEPTABLE"
        else:
            rating = "🚨 SLOW"
            
        print(f"🏆 Performance: {rating}")
    
    # Overall statistics
    print(f"\n{'='*60}")
    print("📊 OVERALL PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    overall_avg = statistics.mean(overall_times)
    overall_min = min(overall_times)
    overall_max = max(overall_times)
    overall_std = statistics.stdev(overall_times)
    
    print(f"📈 Total Runs: {len(overall_times)}")
    print(f"⏱️  Average Construction Time: {overall_avg:.2f}ms")
    print(f"🎯 Range: {overall_min:.2f}ms - {overall_max:.2f}ms")
    print(f"📊 Standard Deviation: ±{overall_std:.2f}ms")
    
    # Throughput calculations
    pags_per_second = 1000 / overall_avg
    pags_per_minute = pags_per_second * 60
    
    print(f"\n🚀 THROUGHPUT ANALYSIS")
    print(f"PAGs per second: {pags_per_second:.1f}")
    print(f"PAGs per minute: {pags_per_minute:.0f}")
    
    # Performance classification
    if overall_avg < 10:
        classification = "🚀 ULTRA-FAST (Suitable for real-time analysis)"
        target_use = "Live game analysis, rapid tournament play"
    elif overall_avg < 50:
        classification = "✅ FAST (Suitable for training and analysis)"
        target_use = "Self-play generation, position analysis"
    elif overall_avg < 100:
        classification = "⚠️  MODERATE (Acceptable for training)"
        target_use = "Batch training, offline analysis"
    else:
        classification = "🚨 SLOW (Optimization needed)"
        target_use = "Limited use, requires optimization"
    
    print(f"\n🏆 PERFORMANCE CLASSIFICATION: {classification}")
    print(f"🎯 Recommended Use: {target_use}")
    
    # Memory and complexity insights
    print(f"\n💾 COMPLEXITY ANALYSIS")
    print(f"PAG system: Python-based with enhanced features")
    print(f"Edge types implemented: 10 different relationship types")
    print(f"Features: Advanced mobility, control, cooperation, king safety, etc.")
    
    return overall_avg, pags_per_second

def benchmark_feature_extraction_breakdown():
    """Benchmark individual components of PAG construction"""
    print(f"\n{'='*60}")
    print("🔍 FEATURE EXTRACTION BREAKDOWN")
    print(f"{'='*60}")
    
    # Test with a complex middlegame position
    fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4"
    board = chess.Board(fen)
    pag_config = PAGConfig()
    
    # Quick overall timing
    times = []
    for _ in range(20):
        start = time.perf_counter()
        pag = PositionalAdjacencyGraph(board, pag_config)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg_time = statistics.mean(times)
    print(f"\n📊 Average total construction time: {avg_time:.2f}ms")
    print(f"🎯 Components included:")
    print(f"   - Piece feature extraction (enhanced mobility, control, attack/defense)")
    print(f"   - Critical square identification (center, king zones, outposts)")
    print(f"   - 10 edge relationship types:")
    print(f"     • Direct relations (piece-to-piece)")
    print(f"     • Control edges (piece-to-square)")
    print(f"     • Mobility edges (movement analysis)")
    print(f"     • Cooperation edges (piece coordination)")
    print(f"     • Obstruction edges (blocking analysis)")
    print(f"     • Vulnerability edges (threat assessment)")
    print(f"     • Pawn structure edges (pawn chains)")
    print(f"     • King safety edges (defense analysis)")
    print(f"     • Center control edges (central dominance)")
    print(f"     • Material tension edges (capture threats)")

def test_rust_pag_availability():
    """Test if the Rust PAG engine is available"""
    print(f"\n{'='*60}")
    print("🔍 RUST PAG ENGINE STATUS")
    print(f"{'='*60}")
    
    try:
        # Try to import the Rust engine
        import rival_ai_engine
        print("✅ Rust PAG engine is available!")
        
        # Test basic functionality
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        start = time.perf_counter()
        features = rival_ai_engine.extract_pag_features(fen)
        end = time.perf_counter()
        rust_time = (end - start) * 1000
        
        print(f"🚀 Rust PAG extraction time: {rust_time:.2f}ms")
        print(f"📊 Features extracted: {len(features.get('pieces', []))} pieces, {len(features.get('critical_squares', []))} critical squares, {len(features.get('edges', []))} edges")
        
        # Compare with Python version
        start = time.perf_counter()
        board = chess.Board(fen)
        pag_config = PAGConfig()
        pag = PositionalAdjacencyGraph(board, pag_config)
        end = time.perf_counter()
        python_time = (end - start) * 1000
        
        print(f"🐍 Python PAG construction time: {python_time:.2f}ms")
        
        if rust_time < python_time:
            speedup = python_time / rust_time
            print(f"🚀 Rust is {speedup:.1f}x faster than Python!")
        else:
            slowdown = rust_time / python_time
            print(f"🐍 Python is {slowdown:.1f}x faster than Rust (unexpected!)")
            
        return True, rust_time
        
    except ImportError:
        print("❌ Rust PAG engine not available")
        print("   This is expected - we're using Python PAG implementation")
        return False, None

if __name__ == "__main__":
    try:
        # Test Rust engine availability
        rust_available, rust_time = test_rust_pag_availability()
        
        # Main benchmark
        avg_time, throughput = benchmark_pag_construction()
        
        # Detailed breakdown
        benchmark_feature_extraction_breakdown()
        
        print(f"\n{'='*60}")
        print("🎉 BENCHMARK COMPLETE!")
        print(f"🚀 Your PAG system can build {throughput:.1f} PAGs per second!")
        print(f"⚡ That's {throughput*60:.0f} PAGs per minute with enhanced features!")
        
        if rust_available:
            rust_throughput = 1000 / rust_time
            print(f"🦀 Rust engine: {rust_throughput:.1f} PAGs/second ({rust_throughput*60:.0f}/minute)")
        
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        print("This might be due to missing dependencies or PAG system issues.")
        import traceback
        traceback.print_exc() 