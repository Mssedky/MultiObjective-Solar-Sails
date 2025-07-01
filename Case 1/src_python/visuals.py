import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def make_plots(ToF, alphaG, gammaG, distances):
    # Plot alphaG vs ToF
    plt.figure(figsize=(10, 6))
    plt.plot(ToF, alphaG)
    plt.plot(ToF, alphaG)
    plt.xlabel('Time of Flight (days)')
    plt.ylabel('Cone Angle (rad)')
    plt.title('Cone Angle Variation Over Time')
    plt.grid(True)
    plt.show()

    # Plot gammaG vs ToF
    plt.figure(figsize=(10, 6))
    plt.plot(ToF, gammaG)
    plt.plot(ToF, gammaG)
    plt.xlabel('Time of Flight (days)')
    plt.ylabel('Clock Angle (rad)')
    plt.title('Clock Angle Variation Over Time')
    plt.grid(True)
    plt.show()


    
    # Plot distances vs ToF
    plt.figure(figsize=(10, 6))
    plt.plot(ToF, distances)
    plt.xlabel('Time of flight (days)')
    plt.ylabel('Distance (AU)')
    plt.title('Distance Between Light sail and NEO Over Time')
    plt.grid(True)
    plt.show()


def make_animation(sunPos, earthPos, NEOPos, sailPos,simTime, NEOname):
    print("Creating animation...")
    # Set up the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

    # To view from above:
    ax.elev = 90
    ax.azim = 0

    # Initialize empty lines for each position
    sun_line, = ax.plot([], [], [], 'yo', markersize=10, label='Sun')
    earth_line, = ax.plot([], [], [], 'bo', markersize=5, label='Earth')

    neo_line, = ax.plot([], [], [], 'go', markersize=5, label='NEO')
    sail_line, = ax.plot([], [], [], 'ro', markersize=3, label='Sail')

    # Update function for animation
    def update(frame):
        ax.clear() 

        # Set labels and title
        ax.set_xlabel('X (AU)')
        ax.set_ylabel('Y (AU)')
        ax.set_zlabel('Z (AU)')
        ax.set_title(f'Optimum Light Sail Trajectory to {NEOname}')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])

        # Plot sun
        sun_line = ax.plot([sunPos[frame][0]], [sunPos[frame][1]], [sunPos[frame][2]], 'yo', markersize=10, label='Sun')[0]

        # Plot Earth
        earth_line = ax.plot([earthPos[frame][0]], [earthPos[frame][1]], [earthPos[frame][2]], 'bo', markersize=3.5, label='Earth')[0]

        # Plot NEO
        neo_line = ax.plot([NEOPos[frame][0]], [NEOPos[frame][1]], [NEOPos[frame][2]], 'go', markersize=2, label='NEO')[0]

        # Plot Sail
        sail_line = ax.plot([sailPos[frame][0]], [sailPos[frame][1]], [sailPos[frame][2]], 'ro', markersize=1, label='Sail')[0]

        # Add text for simulation time
        time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
        time_text.set_text(f"Simulation Time: {simTime[frame]:.2f} days")

        # Show legend
        ax.legend()

        return sun_line, earth_line, neo_line, sail_line, time_text

    # Add text for simulation time
    time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    # Set title
    ax.set_title(f'Optimum Light Sail Trajectory to {NEOname}')

    # Show legend
    ax.legend()

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(simTime), interval=0.4, blit=False)

    # Save the animation
    output_dir = "Case 1/output"
    os.makedirs(output_dir, exist_ok=True)

    ani.save(os.path.join(output_dir, f'Trajectory to {NEOname}.gif'), writer='pillow', fps=12)

    # Show plot
    plt.show()
