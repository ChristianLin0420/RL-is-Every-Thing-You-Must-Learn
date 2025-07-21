"""
Interactive Value Iteration Demo
Author: Day 1 - Bellman Equations & Dynamic Programming Challenge

This script demonstrates the interactive Value Iteration visualization.
You can watch the algorithm step by step as values propagate through the grid!
"""

import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridWorld
from value_iteration import ValueIteration


def main():
    """Run the interactive Value Iteration demo."""
    
    print("🎬" + "="*60)
    print("🎮 INTERACTIVE VALUE ITERATION DEMO")
    print("🎬" + "="*60)
    print()
    print("🎯 This demo shows Value Iteration running step-by-step!")
    print("📺 Watch how values propagate from the goal to all states")
    print("🕹️  Use the interactive controls to explore the algorithm")
    print()
    print("🎮 Controls available:")
    print("   • ▶️ Play/Pause: Auto-play through iterations")
    print("   • ⏮️⏭️ Step: Manual step forward/backward")
    print("   • 🔄 Reset: Go back to iteration 0")
    print("   • 🎚️ Speed: Control animation speed")
    print("   • 📊 Slider: Jump to any iteration")
    print()
    print("🚀 Starting interactive visualization...")
    print("-" * 60)
    
    # Create environment and solver
    env = GridWorld(size=5, gamma=0.9)
    vi = ValueIteration(env, threshold=1e-6, max_iterations=1000)
    
    print(f"📋 Environment: {env.size}×{env.size} GridWorld")
    print(f"🎯 Start: {env.start_state} → Goal: {env.goal_state}")
    print(f"⚙️  Discount γ: {env.gamma}")
    print()
    
    # Run interactive visualization
    try:
        optimal_values, optimal_policy, metrics = vi.solve_interactive(
            animation_speed=800,  # 800ms between iterations
            figsize=(16, 10)
        )
        
        print("\n" + "="*60)
        print("🎉 Interactive demo completed!")
        print(f"✅ Converged in {metrics['iterations']} iterations")
        print(f"📊 Final convergence Δ = {metrics['final_delta']:.2e}")
        print("\n🧠 Key observations:")
        print("   • Values start at 0 and propagate from the goal")
        print("   • Policy arrows point toward higher-value states")
        print("   • Convergence shows exponential decay")
        print("   • The algorithm finds optimal paths to the goal")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("💡 Make sure you have matplotlib with interactive backend")
        print("   Try: pip install matplotlib")


def run_comparison_demo():
    """Run a comparison between different grid sizes."""
    
    print("\n🔄 Running comparison demo with different grid sizes...")
    
    sizes = [3, 4, 5]
    
    for size in sizes:
        print(f"\n📐 Testing {size}×{size} grid...")
        
        env = GridWorld(size=size, gamma=0.9)
        vi = ValueIteration(env, threshold=1e-6)
        
        # Run quick solve to get metrics
        _, _, metrics = vi.solve(verbose=False)
        
        print(f"   ✅ Converged in {metrics['iterations']} iterations")
        print(f"   📊 Final Δ = {metrics['final_delta']:.2e}")
        
        # Show interactive for 5x5 grid
        if size == 5:
            print(f"   🎬 Launching interactive demo for {size}×{size}...")
            vi.solve_interactive(animation_speed=600, figsize=(15, 9))


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Configure matplotlib
    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 100
    
    # Check if interactive backend is available
    try:
        # Test if we can create interactive widgets
        import matplotlib.widgets
        print("✅ Interactive matplotlib backend detected")
        
        # Run the main demo
        main()
        
        # Ask if user wants comparison demo
        print("\n" + "="*60)
        response = input("🤔 Would you like to see comparison demo? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            run_comparison_demo()
        else:
            print("👋 Thanks for trying the interactive Value Iteration demo!")
            
    except ImportError:
        print("❌ Interactive matplotlib widgets not available")
        print("💡 Please install: pip install matplotlib")
        print("🔧 Or use a different backend: plt.switch_backend('TkAgg')")
    except Exception as e:
        print(f"❌ Error setting up interactive demo: {e}")
        print("💡 Try running the regular main.py for static visualizations")
    
    print("\n🎓 Interactive Value Iteration Demo Complete!")
    print("📚 This demonstrates the Bellman optimality equations in action!") 