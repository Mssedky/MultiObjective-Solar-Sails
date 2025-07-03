import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from scipy.optimize import fsolve
from scipy.optimize import root
import time
import pandas as pd
import pprint
from scipy.interpolate import BSpline, make_interp_spline
import sys
import graphviz

from LightSailSolver import *
from LightSail_GA import generate_smooth_angles
from BodiesCalc import *

def visualize_connections(connectionPaths, filename):
    """
    Visualize the tree of connections that deliver data from sails back to Earth
    """

    for i in range(len(connectionPaths)):
        connectionPaths[i][2] = round(connectionPaths[i][2], 1)

    dot = graphviz.Digraph(comment='Network')
    dot.attr('graph', pad="0.5", nodesep="0.6", ranksep="1", dir="back", dpi = "500")

    dot.node('E', 'Earth', shape="box", style="filled", color = "lightgrey", fontsize="25pt")

    i = 0
    current_time = 200
    for connection in connectionPaths:
        connection_index = str(i)
        connection_src = str(connection[1])
        connection_time = str(connection[2])

        pathLoc = connection[3]
        dot.node(connection_index, connection_src, shape="box", style="filled", color = "lightgrey", fontsize="25pt")

        if pathLoc == -1: #pathLoc = -1 means target connection was earth
            dot.edge('E', connection_index, label = ' t = '+connection_time, dir="back")
        else:
            dot.edge(str(pathLoc),connection_index, label = ' t = '+connection_time, dir="back")

        i += 1

    #To pick different color: https://graphviz.org/doc/info/colors.html

    dot.render(filename, format='png', cleanup=True)

def pretty_print_dict(data):
    """
    Pretty prints a dictionary.
    
    Args:
        data (dict): The dictionary to pretty print.
    """
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary.")
    
    printer = pprint.PrettyPrinter(indent=4, width=80, sort_dicts=True)
    printer.pprint(data)


def renderNetwork(timeVec, earthAndSailPos, output_filename, frameInterval = 10, trackNEO = 0, NEOPos = None, connectionPaths = None, twoDim = 0, indexAndTimeofNEOVisit = None):
    print("Making animation...")
    earthPos = earthAndSailPos[0]

    if twoDim == 0: #For 3d plot:
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
        # if trackNEO == 1:
        #     neo_line, = ax.plot([], [], [], 'go', markersize=5, label='NEO')
        sail_line, = ax.plot([], [], [], 'ro', markersize=3, label='Sail')

        # Update function for animation
        def update(frame):
            ax.clear()  # Clear the axes

            # Set labels and title
            ax.set_xlabel('X (AU)')
            ax.set_ylabel('Y (AU)')
            ax.set_zlabel('Z (AU)')
            ax.set_title('Optimum Light Sail Trajectory to NEO 2016 VO1')
            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_zlim([-2, 2])

            # # Plot sun
            # sun_line = ax.plot([sunPos[frame][0]], [sunPos[frame][1]], [sunPos[frame][2]], 'yo', markersize=10, label='Sun')[0]

            # Plot Earth
            earth_line = ax.plot([earthPos[frame][0]], [earthPos[frame][1]], [earthPos[frame][2]], 'bo', markersize=3.5, label='Earth')[0]

            # Plot NEO
            if trackNEO == 1:
                neo_line = ax.plot([NEOPos[frame][0]], [NEOPos[frame][1]], [NEOPos[frame][2]], 'go', markersize=2, label='NEO')[0]

            #Plot Sail
            for i in range(1,len(earthAndSailPos)):
                sail_line = ax.plot([earthAndSailPos[i][frame][0]], [earthAndSailPos[i][frame][1]], [earthAndSailPos[i][frame][2]], 'ro', markersize=1)[0]
                ax.text(earthAndSailPos[i][frame][0], earthAndSailPos[i][frame][1], earthAndSailPos[i][frame][2], str(i))
            # sail_line = ax.plot([earthAndSailPos[i][frame][0]], [earthAndSailPos[i][frame][1]], [earthAndSailPos[i][frame][2]], 'ro', markersize=1, label='Sail')[0]

            #Add legend entry by plotting w no data:
            sail_line = ax.plot([], [], [], 'ro', markersize=1, label='Sail')[0]

            # # Update simulation time text
            # time_text.set_text(f"Simulation Time: {timeVec[frame]:.2f} days")
            # time_text.set_text("test")
            # Add text for simulation time
            time_text = ax.text2D(0.05, 0.95, f"Simulation Time: {timeVec[frame]:.2f} days", transform=ax.transAxes)
            # time_text.set_text(f"Simulation Time: {simTime[frame]:.2f} days")

            # Show legend
            ax.legend()

            # return sun_line, earth_line, neo_line, sail_line, time_text

        # Add text for simulation time
        time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

        # Set title
        ax.set_title('Optimum Light Sail Trajectory to NEO 2016 VO1')

        # Show legend
        ax.legend()

    else: #For 2d plot of xy plane:
        mainfig_x1 = -2
        mainfig_x2 = 2
        mainfig_y1 = -2
        mainfig_y2 = 2

        # Set up the figure and 3D axis
        fig = plt.figure(dpi = 200)
        ax = fig.add_subplot(111)
        axins = ax.inset_axes([1.1, 0.275, 0.47, 0.47], xlim=(mainfig_x1, mainfig_x2), ylim=(mainfig_y1, mainfig_y2), xticklabels=[], yticklabels=[])
        plt.tight_layout()

        # Update function for animation
        def update(frame):
            ax.clear()  # Clear the axes

            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_aspect('equal', adjustable='box')

            # ax.scatter([0],[0], c='y', s = 10)

            # Plot sun
            ax.scatter([0], [0], c = 'y', s=75, label='Sun')

            # Plot Earth position and trail
            ax.scatter([earthPos[frame][0]], [earthPos[frame][1]], c = 'b', s=50, label='Earth')
            if frame < timeVec.shape[0]:
                ax.plot(earthPos[:frame+1, 0], earthPos[:frame+1, 1], c = 'b')

            # Plot NEO
            if trackNEO == 1:
                ax.scatter([NEOPos[frame][0]], [NEOPos[frame][1]], c = 'g', s=30, label='NEO')
                if frame < timeVec.shape[0]:
                    ax.plot(NEOPos[:frame+1, 0], NEOPos[:frame+1, 1], c = 'g')

            # Plot network sails
            for i in range(1,len(earthAndSailPos)):
                # ax.scatter([earthAndSailPos[i][frame][0]], [earthAndSailPos[i][frame][1]], c = 'r', s=15)
                if frame < timeVec.shape[0]:
                    ax.plot(earthAndSailPos[i][:frame+1, 0], earthAndSailPos[i][:frame+1, 1], c = 'r', alpha = 0.1)
                ax.scatter([earthAndSailPos[i][frame][0]], [earthAndSailPos[i][frame][1]], c = 'r', s=15)
                # ax.text(earthAndSailPos[i][frame][0], earthAndSailPos[i][frame][1], str(i))
            ax.scatter([], [], c = 'r', s=15, label = 'Sail')
            
            ax.text(0.025, 0.95, f"Simulation Time: {timeVec[frame]:.2f} days", transform=ax.transAxes)

            # Show legend
            ax.legend(bbox_to_anchor=(1.025, 1), loc='upper left', borderaxespad=0)

            #Turn off axis tick marks and labels
            ax.set_xticks([])
            ax.set_yticks([])


        # Set title
        ax.set_title('Optimum Light Sail Trajectory to NEO 2016 VO1')

        # # Show legend
        # ax.legend()


    # Create animation
    frames=np.arange(0,len(timeVec), frameInterval)
    if connectionPaths is None:
        pass
    else:
        added_frames = np.int64(np.unique(np.array(connectionPaths)[:,2]/dT))
        frames = np.concatenate((frames, added_frames))
        frames = np.unique(frames)
        frames = np.sort(frames)

    ani = FuncAnimation(fig, update, frames=frames, interval=0.4, blit=False)

    # Save the animation
    ani.save(output_filename, writer='pillow', fps=12)

    #-----------------------------------------------
    #---------Connections inset animation-----------
    #-----------------------------------------------

    mainfig_x1 = -2
    mainfig_x2 = 2
    mainfig_y1 = -2
    mainfig_y2 = 2

    # # Set up the figure
    fig = plt.figure(dpi = 200)
    ax = fig.add_subplot(111)
    axins = ax.inset_axes([1.1, 0.275, 0.47, 0.47], xlim=(mainfig_x1, mainfig_x2), ylim=(mainfig_y1, mainfig_y2), xticklabels=[], yticklabels=[])
    plt.tight_layout()

    # ax.set_xlabel('X (AU)')
    # ax.set_ylabel('Y (AU)')

    # Update function for animation
    def connections(frame):
        ax.clear()  # Clear the axes

        obj1_ind = connectionPaths[frame][0]
        obj2_ind = connectionPaths[frame][1]
        time = connectionPaths[frame][2]
        time_ind = np.where(timeVec == time)[0][0]

        xmin = min(earthAndSailPos[obj1_ind][time_ind][0], earthAndSailPos[obj2_ind][time_ind][0])
        xmax = max(earthAndSailPos[obj1_ind][time_ind][0], earthAndSailPos[obj2_ind][time_ind][0])
        x1 = (xmax+xmin)/2-0.1
        x2 = (xmax+xmin)/2+0.1
        ymin = min(earthAndSailPos[obj1_ind][time_ind][1], earthAndSailPos[obj2_ind][time_ind][1])
        ymax = max(earthAndSailPos[obj1_ind][time_ind][1], earthAndSailPos[obj2_ind][time_ind][1])
        y1 = (ymax+ymin)/2-0.1
        y2 = (ymax+ymin)/2+0.1

        axins = ax.inset_axes([1.05, 0.13, 0.47, 0.47], xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])

        ax.set_xlim([mainfig_x1, mainfig_x2])
        ax.set_ylim([mainfig_y1, mainfig_y2])
        ax.set_aspect('equal', adjustable='box')
        axins.set_aspect('equal', adjustable='box')

        # Plot sun
        ax.scatter([0], [0], c = 'y', s=75, label='Sun')
        axins.scatter([0], [0], c = 'y', s=75, label='Sun')

        # Plot Earth position and trail
        ax.scatter([earthPos[time_ind][0]], [earthPos[time_ind][1]], c = 'b', s=50, label='Earth')
        if time_ind < timeVec.shape[0]:
            ax.plot(earthPos[:time_ind+1, 0], earthPos[:time_ind+1, 1], c = 'b')
        axins.scatter([earthPos[time_ind][0]], [earthPos[time_ind][1]], c = 'b', s=50, label='Earth')
        if time_ind < timeVec.shape[0]:
            axins.plot(earthPos[:time_ind+1, 0], earthPos[:time_ind+1, 1], c = 'b')

        # Plot NEO
        if trackNEO == 1:
            ax.scatter([NEOPos[time_ind][0]], [NEOPos[time_ind][1]], c = 'g', s=30, label='NEO')
            if time_ind < timeVec.shape[0]:
                ax.plot(NEOPos[:time_ind+1, 0], NEOPos[:time_ind+1, 1], c = 'g')
        if trackNEO == 1:
            axins.scatter([NEOPos[time_ind][0]], [NEOPos[time_ind][1]], c = 'g', s=30, label='NEO')
            if time_ind < timeVec.shape[0]:
                axins.plot(NEOPos[:time_ind+1, 0], NEOPos[:time_ind+1, 1], c = 'g')

        # Plot network sails
        for i in range(1,len(earthAndSailPos)):
            if time_ind < timeVec.shape[0]:
                ax.plot(earthAndSailPos[i][:time_ind+1, 0], earthAndSailPos[i][:time_ind+1, 1], c = 'r', alpha = 0.1)
            ax.scatter([earthAndSailPos[i][time_ind][0]], [earthAndSailPos[i][time_ind][1]], c = 'r', s=15)
            # if mainfig_x1 < earthAndSailPos[i][time_ind][0] and mainfig_x2 > earthAndSailPos[i][time_ind][0] and mainfig_y1 < earthAndSailPos[i][time_ind][1] and mainfig_y2 > earthAndSailPos[i][time_ind][1]:
            #     ax.text(earthAndSailPos[i][time_ind][0], earthAndSailPos[i][time_ind][1], str(i))
        ax.scatter([], [], c = 'r', s=15, label = 'Sail')
        for i in range(1,len(earthAndSailPos)):
            if time_ind < timeVec.shape[0]:
                axins.plot(earthAndSailPos[i][:time_ind+1, 0], earthAndSailPos[i][:time_ind+1, 1], c = 'r', alpha = 0.1)
            axins.scatter([earthAndSailPos[i][time_ind][0]], [earthAndSailPos[i][time_ind][1]], c = 'r', s=15)
            if x1 < earthAndSailPos[i][time_ind][0] and x2 > earthAndSailPos[i][time_ind][0] and y1 < earthAndSailPos[i][time_ind][1] and y2 > earthAndSailPos[i][time_ind][1]:
                axins.text(earthAndSailPos[i][time_ind][0], earthAndSailPos[i][time_ind][1], str(i))
        axins.scatter([], [], c = 'r', s=15, label = 'Sail')
        
        ax.text(0.025, 0.95, f"Simulation Time: {timeVec[time_ind]:.2f} days", transform=ax.transAxes)

        ax.indicate_inset_zoom(axins, edgecolor="black")

        ax.legend(bbox_to_anchor=(1.025, 1), loc='upper left', borderaxespad=0)

        #Turn off axis tick marks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        axins.set_xticks([])
        axins.set_yticks([])

    # Create animation
    frames=np.arange(0,len(connectionPaths))

    ani = FuncAnimation(fig, connections, frames=frames, interval=0.4, blit=False)

    # Save the animation
    ani.save("connections_"+output_filename, writer='pillow', fps=12)

    #-----------------------------------------------
    #---------NEO min dist figure-----------
    #-----------------------------------------------

    if indexAndTimeofNEOVisit is not None:

        for i in range(len(indexAndTimeofNEOVisit)):

            mainfig_x1 = -2
            mainfig_x2 = 2
            mainfig_y1 = -2
            mainfig_y2 = 2

            # # Set up the figure
            fig = plt.figure(dpi = 500)
            ax = fig.add_subplot(111)
            axins = ax.inset_axes([1.1, 0.275, 0.47, 0.47], xlim=(mainfig_x1, mainfig_x2), ylim=(mainfig_y1, mainfig_y2), xticklabels=[], yticklabels=[])
            plt.tight_layout()

            ax.clear()  # Clear the axes

            obj1_ind = indexAndTimeofNEOVisit[i][0]
            time = indexAndTimeofNEOVisit[i][1]
            time_ind = np.where(timeVec == time)[0][0]

            x = earthAndSailPos[obj1_ind][time_ind][0]
            x1 = x-0.1
            x2 = x+0.1
            y = earthAndSailPos[obj1_ind][time_ind][1]
            y1 = y-0.1
            y2 = y+0.1

            axins = ax.inset_axes([1.05, 0.13, 0.47, 0.47], xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])

            ax.set_xlim([mainfig_x1, mainfig_x2])
            ax.set_ylim([mainfig_y1, mainfig_y2])
            ax.set_aspect('equal', adjustable='box')
            axins.set_aspect('equal', adjustable='box')

            # Plot sun
            ax.scatter([0], [0], c = 'y', s=75, label='Sun')
            axins.scatter([0], [0], c = 'y', s=75, label='Sun')

            # Plot Earth position and trail
            ax.scatter([earthPos[time_ind][0]], [earthPos[time_ind][1]], c = 'b', s=50, label='Earth')
            if time_ind < timeVec.shape[0]:
                ax.plot(earthPos[:time_ind+1, 0], earthPos[:time_ind+1, 1], c = 'b')
            axins.scatter([earthPos[time_ind][0]], [earthPos[time_ind][1]], c = 'b', s=50, label='Earth')
            if time_ind < timeVec.shape[0]:
                axins.plot(earthPos[:time_ind+1, 0], earthPos[:time_ind+1, 1], c = 'b')

            # Plot NEO
            if trackNEO == 1:
                ax.scatter([NEOPos[time_ind][0]], [NEOPos[time_ind][1]], c = 'g', s=30, label='NEO')
                if time_ind < timeVec.shape[0]:
                    ax.plot(NEOPos[:time_ind+1, 0], NEOPos[:time_ind+1, 1], c = 'g')
            if trackNEO == 1:
                axins.scatter([NEOPos[time_ind][0]], [NEOPos[time_ind][1]], c = 'g', s=30, label='NEO')
                if time_ind < timeVec.shape[0]:
                    axins.plot(NEOPos[:time_ind+1, 0], NEOPos[:time_ind+1, 1], c = 'g')

            # Plot network sails
            for i in range(1,len(earthAndSailPos)):
                if time_ind < timeVec.shape[0]:
                    ax.plot(earthAndSailPos[i][:time_ind+1, 0], earthAndSailPos[i][:time_ind+1, 1], c = 'r', alpha = 0.1)
                ax.scatter([earthAndSailPos[i][time_ind][0]], [earthAndSailPos[i][time_ind][1]], c = 'r', s=15)
                # if mainfig_x1 < earthAndSailPos[i][time_ind][0] and mainfig_x2 > earthAndSailPos[i][time_ind][0] and mainfig_y1 < earthAndSailPos[i][time_ind][1] and mainfig_y2 > earthAndSailPos[i][time_ind][1]:
                #     ax.text(earthAndSailPos[i][time_ind][0], earthAndSailPos[i][time_ind][1], str(i))
            ax.scatter([], [], c = 'r', s=15, label = 'Sail')
            for i in range(1,len(earthAndSailPos)):
                if time_ind < timeVec.shape[0]:
                    axins.plot(earthAndSailPos[i][:time_ind+1, 0], earthAndSailPos[i][:time_ind+1, 1], c = 'r', alpha = 0.1)
                axins.scatter([earthAndSailPos[i][time_ind][0]], [earthAndSailPos[i][time_ind][1]], c = 'r', s=15)
                if x1 < earthAndSailPos[i][time_ind][0] and x2 > earthAndSailPos[i][time_ind][0] and y1 < earthAndSailPos[i][time_ind][1] and y2 > earthAndSailPos[i][time_ind][1]:
                    axins.text(earthAndSailPos[i][time_ind][0], earthAndSailPos[i][time_ind][1], str(i))
            axins.scatter([], [], c = 'r', s=15, label = 'Sail')
            
            ax.text(0.025, 0.95, f"Simulation Time: {timeVec[time_ind]:.2f} days", transform=ax.transAxes)

            ax.indicate_inset_zoom(axins, edgecolor="black")

            ax.legend(bbox_to_anchor=(1.025, 1), loc='upper left', borderaxespad=0)

            #Turn off axis tick marks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            axins.set_xticks([])
            axins.set_yticks([])

            plt.savefig(("NEOmindist"+str(i)+"_"+output_filename).replace(".gif", ".png"))





class lightSailNetworkSolver:

    def __init__(self):
        # self.var = var
        # self.bodies = bodies
        pass

    def checkForBranches(self, unterminatedBranchConnections, unprocessedConnections, newTarget, pathLoc, time, dT):
        '''
        unterminatedBranchConnections: list with format [(target, source, time, pathLoc), ...] containing connections from branches that haven't been terminated yet (queue to process and find branches)
        unprocessedConnections: list with format [(target, source, time), (target, source, time)...] containing unprocessed connections
        newTarget: index of target object, will find connections with newTarget as target
        pathLoc: index of parent path in connectionPaths
        time: time of parent connection. the branching connection will need to have earlier time to create a connected path
        '''
        potentialBranchConnections = unprocessedConnections[str(int(newTarget))].copy()
        #check for consecutive connections, keep only last one if there are consecutive connections
        #assume connections are listed in order of ascending time, as they would be based on code above

        i = 0
        while i < len(potentialBranchConnections)-1:
            # print("i", i)
            # print("len(connectedSources)", len(potentialBranchConnections))
            (branchTarget, branchSource, branchTime) = potentialBranchConnections[i]
            (nextConnectedTarget, nextConnectedSource, nextConnectedTime) = potentialBranchConnections[i+1]
            if branchTime <= time:
                if i < len(potentialBranchConnections)-1: #(if not on the last connection)

                    #check if next connection is consecutive in time with same object and still within correct time range:
                    if branchSource == nextConnectedSource and abs(dT-(nextConnectedTime-branchTime)) < dT*0.01 and nextConnectedTime<=time:
                        consecutive = 1
                    else:
                        consecutive = 0

                    #if consecutive, continue stepping through until reach last in consecutive series
                    while consecutive == 1: 
                        #remove the connection from main library:
                        unprocessedConnections[str(int(branchTarget))].remove((branchTarget, branchSource, branchTime))
                        unprocessedConnections[str(int(branchSource))].remove((branchSource, branchTarget, branchTime))
                        
                        #check if next connection also consecutive:
                        i += 1
                        (branchTarget, branchSource, branchTime) = potentialBranchConnections[i]
                        if i < len(potentialBranchConnections)-1: #(if not on the last connection)
                            (nextConnectedTarget, nextConnectedSource, nextConnectedTime) = potentialBranchConnections[i+1]
                            if branchTime <= time:
                                if branchSource == nextConnectedSource and abs(dT-(nextConnectedTime-branchTime)) < dT*0.01 and nextConnectedTime<=time:
                                    consecutive = 1
                                else:
                                    consecutive = 0
                        else:
                            consecutive = 0

                # print(unprocessedConnections[str(int(branchSource))])
                # print("appending path:", (branchTarget, branchSource, branchTime, pathLoc))
                # print("i", i)
                unterminatedBranchConnections.append((branchTarget, branchSource, branchTime, pathLoc)) #save connection that branches from higher level connection
                unprocessedConnections[str(int(branchSource))].remove((branchSource, branchTarget, branchTime)) #remove the connection from main library of unprocessed connetions
                unprocessedConnections[str(int(branchTarget))].remove((branchTarget, branchSource, branchTime)) #remove the connection from main library of unprocessed connetions
                # print(unprocessedConnections[str(int(branchSource))], "\n")

            i += 1

        return unprocessedConnections, unterminatedBranchConnections

    def findConnections(self, numSails, timeVec, earthAndSailPos, distToConnect, takeOffInd, unprocessedConnections, connectionDelay, dT, T):
        delayedSailActive = np.zeros((numSails+1, timeVec.shape[0]))
        delayedSailActive[0,:] = np.ones(timeVec.shape[0])
        for i in range(1,numSails+1):
            delayedSailActive[i,int(takeOffInd[i]+connectionDelay/dT):] = 1

        dist_matrix = np.zeros((numSails+1, numSails+1, timeVec.shape[0]))
        # minDistToEarth = np.inf
        for i in range(numSails+1):
            for j in range(i+1,numSails+1):
                # print("calculating dists", i, j)
                dist = earthAndSailPos[i]-earthAndSailPos[j]
                dist = np.sqrt(dist[:,0]**2 + dist[:,1]**2 + dist[:,2]**2)
                dist_matrix[i, j, :] = dist
                # connections = np.where((dist < distToConnect) & (earthAndSailPos[i][:,3]*earthAndSailPos[j][:,3]==1)) #both objects within distance and active (no connectionDelay)
                connections = np.where((dist < distToConnect) & (delayedSailActive[i,:]*delayedSailActive[j,:]==1)) #both objects within distance and active (with connectionDelay)
                temp = [(j, i, timeVec[x]) for x in connections[0]] #entry with (target, source, time)
                unprocessedConnections[str(int(j))] = unprocessedConnections[str(int(j))] + temp
                temp = [(i, j, timeVec[x]) for x in connections[0]]
                unprocessedConnections[str(int(i))] = unprocessedConnections[str(int(i))] + temp
                
        dist_matrix = dist_matrix + np.transpose(dist_matrix, axes = (1,0,2))

        #find connections to Earth:
        unterminatedBranchConnections = []
        newTarget = 0
        pathLoc = -1
        time = T
        unprocessedConnections, unterminatedBranchConnections = self.checkForBranches(unterminatedBranchConnections, unprocessedConnections, newTarget, pathLoc, time, dT)

        connectionPaths = []
        endCond = 0
        while len(unterminatedBranchConnections) > 0:
            con = unterminatedBranchConnections[0]
            target = con[0] #on first loop will all be 0 (Earth)
            source = con[1] 
            time = con[2]
            pathLoc = con[3] #index of parent connection in connectionPaths

            #Save connection path
            connectionPaths.append([target, source, time, pathLoc])
            pathLoc = len(connectionPaths)-1

            newTarget = source

            unprocessedConnections, unterminatedBranchConnections = self.checkForBranches(unterminatedBranchConnections, unprocessedConnections, newTarget, pathLoc, time, dT)

            unterminatedBranchConnections.pop(0)

        return dist_matrix, connectionPaths

    

    def runNetworkSim(self, pop, earthPos, optimizedPathPos, dT, T, indexAndTimeofNEOVisit, prevEarthAndSailPos = None, toRerun = None, render = 0, renderFrameInterval = 10, trackNEO = 0, NEOPos = None, w = None, output_filename = None):
        """
        prevEarthAndSailPos: EarthAndSailPos from a previous run. if None, then all sails are run to determine their positions over time. If given, sails are rerun according to argument "toRerun"
        toRerun: list of indices of network sails to be rerun (indexing is on the optimizable network sails contained in pop -- index 0 reruns the sail given by design pop[0,:])
        w: weights for cost function, only needed if rendering (required to run calcNetworkCost)
        """
        numNetworkSails = pop.shape[0]
        earthAndSailPos = [earthPos] #all earth pos and network sail pos data for pre-optimized and other sails
        earthAndSailPos = earthAndSailPos + optimizedPathPos
        numNonNetworkObjects = len(earthAndSailPos)
        numSeg = int((pop[0].shape[0]-3)/3)

        if prevEarthAndSailPos is None:
            #run simulation for each network sail to get sailPos:
            self.minToSun = [] #closest distance between sun and sail
            for j in range(numNetworkSails):
                var = pop[j,:]
                solver = lightSailSolver(var,[Earth])
                solver.runSim(desHoverTime = 0, constant_angles = 0, T = T, TOL = TOLNEO, TOL2 = TOLEarth, MakeMovie = 0, NumSeg=numSeg, dT=dT, trackNEO = 0, useEndCond = 0)
                earthAndSailPos.append(np.column_stack((solver.sailPos,solver.sailActive)))
                self.minToSun.append(np.min(solver.sailSunDistArray))
        else:
            for j in toRerun:
                earthAndSailPos = prevEarthAndSailPos
                var = pop[j,:]
                solver = lightSailSolver(var,[Earth])
                solver.runSim(desHoverTime = 0, constant_angles = 0, T = T, TOL = TOLNEO, TOL2 = TOLEarth, MakeMovie = 0, NumSeg=numSeg, dT=dT, trackNEO = 0, useEndCond = 0)

                earthAndSailPos[numNonNetworkObjects+j,:,:] = np.column_stack((solver.sailPos,solver.sailActive))
                self.minToSun[j] = np.min(solver.sailSunDistArray)

        timeVec = solver.simTime
        self.earthAndSailPos = earthAndSailPos
        self.timeVec = timeVec

        if render == 1:
            if output_filename is None:
                print("No output_filename given for runNetworkSim")
            else:
                self.calcNetworkCost(pop, earthPos, optimizedPathPos, dT, T, indexAndTimeofNEOVisit, w, interp = 0, dTFine = 0.1, prevEarthAndSailPos = earthAndSailPos, toRerun = [], runNetworkSim = 0)
                renderNetwork(timeVec, earthAndSailPos, output_filename+"_animation.gif", renderFrameInterval, trackNEO, NEOPos, connectionPaths = self.connectionPaths, twoDim = 1, indexAndTimeofNEOVisit = indexAndTimeofNEOVisit)
                if w is None: 
                    print("Can't generate connections tree visualization because input w to runNetworkSim is None")
                visualize_connections(self.connectionPaths, output_filename+"_connections")

        return earthAndSailPos, timeVec


    def calcNetworkCost(self, pop, earthPos, optimizedPathPos, dT, T, indexAndTimeofNEOVisit, w, interp = 0, dTFine = 0.1, prevEarthAndSailPos = None, toRerun = None, runNetworkSim = 1):
        '''
        pop is matrix with dimensions (numNetworkSails, dv)

        assumes that all paths will have same time step and overall simulation time
        assume there is at least one "network" sail
        indexAndTimeofNEOVisit: list with entries in the format: [index of object that visited NEO, earliest time of visit to NEO]
        interp: gives option to interpolate object locations through time with dTFine for calculating distances and finding connections
        prevEarthAndSailPos: can pass in original_earthAndSailPos from a previous run, will use that instead of rerunning all sail objects again
        toRerun: list. if have passed prevEarthAndSailPos, will only rerun network sails with index given in list
        w: list of weights for cost function
        runNetworkSim: if network sim has not already been run, use 1 to run. otherwise assumes that it has already been run and pulls results from self.earthAndSailPos and self.timeVec
        '''

        # -----------------------------------------
        # ---------Run network simulation----------
        # -----------------------------------------
        numNetworkSails = pop.shape[0]
        numOptimizedPaths = len(optimizedPathPos)
        numSails = numOptimizedPaths + numNetworkSails
        unprocessedConnections = {str(i): [] for i in range(0, numSails + 1)} #initialize dictionary with identifiers "0", "1", ..."numSail+1". entry will be list of [(source, time), (source, time)...] for each connection with corresponding identifier as target
        if runNetworkSim == 1:
            earthAndSailPos, timeVec = self.runNetworkSim(pop, earthPos, optimizedPathPos, dT, T, indexAndTimeofNEOVisit, prevEarthAndSailPos = prevEarthAndSailPos, toRerun = toRerun)
        else:
            earthAndSailPos = self.earthAndSailPos 
            timeVec = self.timeVec
        original_earthAndSailPos = np.copy(earthAndSailPos)

        # ------------------------------------------------
        # ---------Calculate network connections----------
        # ------------------------------------------------
        timeSteps = optimizedPathPos[0].shape[0]
        distToConnect = 10000000e3/AU  #AU (10 million km)

        if interp == 1:
            dTFine = 0.1
            timeVecFiner = np.linspace(timeVec[0], timeVec[-1], int(timeVec[-1]/dTFine)+1)
            dTFine = timeVecFiner[1]-timeVecFiner[0]
            earthAndSailPos = np.swapaxes(earthAndSailPos, 0, 1)
            spl = make_interp_spline(timeVec, earthAndSailPos, k = 1) #degree k = 1
            earthAndSailPos = spl(timeVecFiner)
            earthAndSailPos = np.swapaxes(earthAndSailPos, 0, 1)
            timeVec = timeVecFiner
            dT = dTFine

        takeOffInd = []
        for i in range(numSails+1):
            takeOffInd.append(np.searchsorted(earthAndSailPos[i][:,3],1))

        connectionDelay = 100 #number of days after take off until connections can be established (prevents immediate trivial connections to Earth)

        sentBackInfoFromNEO_list = []
        minInfoReturnTime_list = []
        # for distToConnect in distToConnect_tests:
        dist_matrix, connectionPaths = self.findConnections(numSails, timeVec, earthAndSailPos, distToConnect, takeOffInd, unprocessedConnections, connectionDelay, dT, T)
        
        for i in range(len(indexAndTimeofNEOVisit)):
            print("checking connection for indexAndTimeofNEOVisit:", indexAndTimeofNEOVisit[i])
            #check connectionPaths for connection where source is object, time is later than NEO visit
            NEOVisitObjectInd = indexAndTimeofNEOVisit[i][0]
            NEOVisitTime = indexAndTimeofNEOVisit[i][1]
            sentBackInfoFromNEO = 0
            minInfoReturnTime = T #minimum time for info from NEO visit to return to Earth
            for j in range(len(connectionPaths)):
                if connectionPaths[j][1] == NEOVisitObjectInd and connectionPaths[j][2] >= NEOVisitTime:
                    pathLocVal = connectionPaths[j][3]
                    nextPathLocVal = connectionPaths[pathLocVal][3]
                    while nextPathLocVal != -1:
                        pathLocVal = connectionPaths[pathLocVal][3]
                        nextPathLocVal = connectionPaths[pathLocVal][3]

                    minInfoReturnTime = min(connectionPaths[pathLocVal][2], minInfoReturnTime)

                    sentBackInfoFromNEO=1

            sentBackInfoFromNEO_list.append(sentBackInfoFromNEO)
            minInfoReturnTime_list.append(minInfoReturnTime)

        # dist_matrix, connectionPaths = self.findConnections(numSails, timeVec, earthAndSailPos, distToConnect, takeOffInd, unprocessedConnections, connectionDelay, dT)
        print('sentBackInfoFromNEO_list', sentBackInfoFromNEO_list)
        print('minInfoReturnTime_list', minInfoReturnTime_list)

        # minDistToEarth = np.min(dist_matrix[0,1+numOptimizedPaths:,:])
        
        minDistToSailGoingToNEO = 0 #total of min distances between a sail after it has visited NEO and network sails with optimizable paths
        closestNetworkSailInds = []
        for objInd, NEOVisitTime in indexAndTimeofNEOVisit:
            minDistToSailGoingToNEO += np.min(dist_matrix[objInd, numOptimizedPaths+1:numSails+1, int(NEOVisitTime/dT):])
            closestNetworkSailInd = np.argmin(np.min(dist_matrix[objInd, numOptimizedPaths+1:numSails+1, int(NEOVisitTime/dT):], axis = 1)) + numOptimizedPaths+1
            closestNetworkSailInds.append(closestNetworkSailInd)

        #calculate minimum distances of every sail to any other sail after time of first NEO visit:
        min_dists = 0
        max_dists = 0
        minmax_dist = np.inf
        timeOfFirstNEOVisit = np.min(np.array(indexAndTimeofNEOVisit)[:,1])
        for i in range(numOptimizedPaths+1,numSails+1):
            fill_matrix = np.zeros((numSails-numOptimizedPaths, timeVec[int(timeOfFirstNEOVisit/dT):].shape[0]))
            fill_matrix[i-(numOptimizedPaths+1), :] = np.inf
            min_dists += np.min(dist_matrix[i, numOptimizedPaths+1:, int(timeOfFirstNEOVisit/dT):]+fill_matrix)
            max_dists += np.max(dist_matrix[i, numOptimizedPaths+1:, int(timeOfFirstNEOVisit/dT):]-fill_matrix)
            minmax_dist = min(minmax_dist, np.max(dist_matrix[i, numOptimizedPaths+1:, int(timeOfFirstNEOVisit/dT):]-fill_matrix))

        #calculate minimum distances of every sail to any other sail after some amount of time after takeoff for each sail:
        delayToCheckDist = 150 # days (time after takeoff to start checking for connections)
        takeOffIndMatrix = np.zeros((numSails+1, numSails+1))
        for i in range(numSails+1):
            for j in range(numSails+1):
                takeOffIndMatrix[i,j] = max(takeOffInd[i],takeOffInd[j])
        min_dists_w_delay = 0
        minDistToEarth = np.inf
        for i in range(numOptimizedPaths+1,numSails+1):
            if np.min(dist_matrix[i, 0, int(takeOffIndMatrix[i,i]+delayToCheckDist/dT):]) < minDistToEarth:
                minDistToEarth = np.min(dist_matrix[i, 0, int(takeOffIndMatrix[i,i]+delayToCheckDist/dT):])
                closestToEarthInd = i
            min_temp = np.inf
            for j in range(numOptimizedPaths+1,numSails+1):
                if i != j:
                    if np.min(dist_matrix[i, j, int(takeOffIndMatrix[i,j]+delayToCheckDist/dT):]) < min_temp:
                        min_temp = np.min(dist_matrix[i, j, int(takeOffIndMatrix[i,j]+delayToCheckDist/dT):])
            min_dists_w_delay += min_temp

        print("Connection paths", connectionPaths)

        #Calculate cost

        w1 = w[0] #end goal of getting info from NEO visits back to Earth -- should be >=1
        w2 = w[1] #number of connections
        w3 = w[2] #min distance out of all network sails to earth
        w4 = w[3] #min distance out of all network sails to the sail going to NEO
        w5 = w[4] #sum of min dist between network sails and every other network sail
        w6 = w[5] #minmax dist between network sails and every other network sail
        w7 = w[6] #penalty for flying too close to the sun

        print("sentBackInfoFromNEO_list", sentBackInfoFromNEO_list)
        print("minInfoReturnTime_list", minInfoReturnTime_list)
        print("T", T)
               # endGoal = w1*(1-np.sum(sentBackInfoFromNEO_list)/len(indexAndTimeofNEOVisit))**3 #assuming that len(indexAndTimeofNEOVisit) is number of total sails that have visited a NEO, or num of total separate NEO visits
        endGoal = 1
        for i in range(len(sentBackInfoFromNEO_list)):
            if sentBackInfoFromNEO_list[i] == 1:
                endGoal = endGoal*(1/w1)*(minInfoReturnTime_list[i]/T)

        sunClose = min(self.minToSun) #closest distance of any network sail to sun
        sunCost = 0 if sunClose > 0.25 else w7

        if np.sum(sentBackInfoFromNEO_list) == len(indexAndTimeofNEOVisit): #assuming that len(indexAndTimeofNEOVisit) is number of total sails that have visited a NEO, or num of total separate NEO visits
            cost = endGoal
        else:
            cost = endGoal*(w2*1/(len(connectionPaths)+1) + (w3*minDistToEarth)*(w4*minDistToSailGoingToNEO) + w5*(min_dists)+w6*np.exp(-minmax_dist)+w7*sunCost)
            
        print("cost from endGoal:", endGoal)
        print("cost from connections:", 1/(len(connectionPaths)+1))
        print("cost from minDistToEarth:", minDistToEarth)
        print("cost from minDistToSailGoingToNEO:", minDistToSailGoingToNEO)
        print("number of NEO visits with info sent back to Earth", np.sum(sentBackInfoFromNEO_list))
        print("sum of min dist between network sails", min_dists)
        print("sum of max dist between network sails", max_dists)
        print("minimum max dist between network sails", minmax_dist)
        print("min dists cost component", w5*(min_dists))
        print("minmax dists cost component", w6*np.exp(-minmax_dist))
        print("total cost:", cost)

        self.cost = cost
        self.earthAndSailPos = original_earthAndSailPos
        if interp == 1:
            self.interpEarthAndSailPos = earthAndSailPos
        else:
            self.interpEarthAndSailPos = None
        self.connectionPaths = connectionPaths
        self.closestNetworkSailInds = closestNetworkSailInds
        self.closestToEarthInd = closestToEarthInd

        return cost, original_earthAndSailPos

def GeneticAlgorithm(S,P,K,TOL,G,dv,lb,ub, w, func,max_variation, numSeg, numNetworkSails, optimizedPathPos, earthPos, indexAndTimeofNEOVisit, dT, T, render = 0):
    cost=np.ones((S,1))*1000
    prev_cost = None
    Pi=np.empty((0,S))
    meanParents=[]
    Orig=np.zeros((G,S))
    Children=np.zeros((K,numNetworkSails,dv))
    Parents=np.zeros((P,numNetworkSails,dv))
    Lambda=np.zeros((S,numNetworkSails,dv))
    Gen=1
    start = 0

    numOptimizedPaths = len(optimizedPathPos)
    numSails = numOptimizedPaths + numNetworkSails
    
    # Generate starting population
    pop_new = np.zeros((S, numNetworkSails, dv))
    for i in range(S):
        for j in range(numNetworkSails):
            pop_new[i, j, :3+numSeg] = np.random.uniform(lb[:3+numSeg], ub[:3+numSeg])
            pop_new[i, j, 3+numSeg:3+2*numSeg] = generate_smooth_angles(numSeg, lb[3+numSeg:], ub[3+numSeg:], max_variation) #cone angle generation
            pop_new[i, j, 3+2*numSeg:3+3*numSeg] = generate_smooth_angles(numSeg, lb[3+2*numSeg:], ub[3+2*numSeg:], max_variation) #clock angle generation
        
    pop_new[:, :, 0] = np.random.randint(lb[0], ub[0] + 1, (S, numNetworkSails))  # Ensure degree is an integer
    earthAndSailPosList = np.zeros((S,numSails+1,earthPos.shape[0],4))

    while np.abs(np.min(cost))>TOL and Gen<G:
        print("**********************************")
        print("Generation number : ", Gen)
        pop=pop_new

        #Evaluate population fitness
        for i in range(start, S):
            print("Generation number : ", Gen)
            print("String : ",i+1)
            solver = lightSailNetworkSolver()
            solver.runNetworkSim(pop[i,:,:], earthPos, optimizedPathPos, dT, T, indexAndTimeofNEOVisit,  prevEarthAndSailPos = None, toRerun = None)
            cost[i], earthAndSailPos = solver.calcNetworkCost(pop[i,:,:], earthPos, optimizedPathPos, dT, T, indexAndTimeofNEOVisit, w, interp = 1, dTFine = 0.1, prevEarthAndSailPos = None, toRerun = None, runNetworkSim = 0)
            # cost[i], earthAndSailPos = func(pop[i,:,:], earthPos, optimizedPathPos, dT, indexAndTimeofNEOVisit, w)
            earthAndSailPosList[i,:,:,:] = earthAndSailPos

        #Sort population fitnesses
        Index=np.argsort(cost[:,0])
        pop=pop[Index,:]
        cost=cost[Index,:]
        earthAndSailPosList = np.array(earthAndSailPosList)[Index]
        print("earthAndSailPosList shape", earthAndSailPosList.shape)

        print(f"Best cost for generation {Gen} : {cost[0]}")
        # print(f"Best cost occurs at initial launch time= {pop[0,0]} days and velocity={pop[0,1]*1.496e11} m/s")

        #Select parents
        Parents=pop[0:P+1,:, :]
        meanParents.append(np.mean(cost[0:P+1]))

        #Generate K offspring
        for i in range(0,K,2):
            #Breeding parents
            alpha=np.random.uniform(0,1)
            beta=np.random.uniform(0,1)
            Children[i, :, :] = Parents[i, :, :] * alpha + Parents[i + 1, :, :] * (1 - alpha)
            Children[i + 1, :, :] = Parents[i, :, :] * beta + Parents[i + 1, :, :] * (1 - beta)

        #Overwrite population with P parents, K children, and S-P-K random values
        random_values = np.random.uniform(lb, ub, (S - P - K, numNetworkSails, dv))
        random_values[:, :, 0] = np.random.randint(lb[0], ub[0] + 1, (S - P - K, numNetworkSails))  # Ensure degree is an integer
        pop_new = np.vstack((Parents, Children, random_values))

        #Store costs and indices for each generation
        Pi= np.vstack((Pi, cost.T))
        Orig[Gen,:]=Index
        #Increment generation counter
        Gen=Gen+1
        start = P

        if args.outputFilename != None:
            np.savetxt(args.outputFilename + "_currentCosts.txt", cost, delimiter=",")
            np.save(args.outputFilename + "_currentDesigns.npy", pop)
        
        if render == 1:
            solver = lightSailNetworkSolver()
            solver.runNetworkSim(pop[0,:,:], earthPos, optimizedPathPos, dT, T, indexAndTimeofNEOVisit, prevEarthAndSailPos = None, toRerun = None, render = 1, w = w, output_filename = args.outputFilename)

        prev_earthAndSailPosList = earthAndSailPosList

    #Store best population 
    Lambda=pop    
    meanPi=np.mean(Pi,axis=1)
    minPi=np.min(Pi,axis=1)
    return Lambda, Pi, Orig, meanPi, minPi, meanParents, cost

def bruteForceOptimization(var, w, dT, T, optimizedPathPos, earthPos, indexAndTimeofNEOVisit, lb, ub, max_iter=20, tolerance=150, m=0.2, printStatements = 1, saveEvery = 50, costTolerance = 0):
    '''
    var: design to improve
    saveEvery: save after every "saveEvery" tested changes to variable
    costTolerance: if reaches costTolerance, end optimization
    '''

    numNetworkSails = var.shape[0]
    dv = var.shape[1]

    solver = lightSailNetworkSolver()
    inc_var_solver = lightSailNetworkSolver()
    dec_var_solver = lightSailNetworkSolver()

    cost, earthAndSailPos = solver.calcNetworkCost(var, earthPos, optimizedPathPos, dT, T, indexAndTimeofNEOVisit, w, interp = 0, dTFine = 0.1)
    print("closest network sail:", solver.closestNetworkSailInds)
    inc_var_solver.minToSun = solver.minToSun
    dec_var_solver.minToSun = solver.minToSun
    prevEarthAndSailPos = earthAndSailPos

    costs = []

    # cost = lightSailCost(var, desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies)
    no_change = 0
    num_changes = 0
    for j in range(max_iter):
        for i in range(0,numNetworkSails):
            for k in range(3,dv):
                if no_change > tolerance:
                    m = m * 0.5
                    no_change = 0

                inc_var = np.copy(var)
                inc_var[i, k] = min(inc_var[i, k] + inc_var[i, k] * m, ub[k])
                dec_var = np.copy(var)
                dec_var[i, k] = max(dec_var[i, k] - dec_var[i, k] * m, lb[k])

                inc_var_cost, inc_var_earthAndSailPos = inc_var_solver.calcNetworkCost(inc_var, earthPos, optimizedPathPos, dT, T, indexAndTimeofNEOVisit, w, interp = 0, dTFine = 0.1, prevEarthAndSailPos = prevEarthAndSailPos, toRerun = [i])
                dec_var_cost, dec_var_earthAndSailPos = dec_var_solver.calcNetworkCost(dec_var, earthPos, optimizedPathPos, dT, T, indexAndTimeofNEOVisit, w, interp = 0, dTFine = 0.1, prevEarthAndSailPos = prevEarthAndSailPos, toRerun = [i])

                if cost > inc_var_cost:
                    var = inc_var
                    cost = inc_var_cost
                    prevEarthAndSailPos = inc_var_earthAndSailPos
                    no_change = 0
                else:
                    no_change += 1

                if cost > dec_var_cost:
                    var = dec_var
                    cost = dec_var_cost
                    prevEarthAndSailPos = dec_var_earthAndSailPos
                    no_change = 0
                else:
                    no_change += 1
                
                if printStatements == 1:
                    print("var #:", i, "m", m, "new cost:", cost)

                costs.append(cost)
                num_changes +=1
                if num_changes%saveEvery == 0:
                    if args.outputFilename != None:
                        np.savetxt(args.outputFilename + "_bruteForceCurrentCosts.txt", np.array(costs), delimiter=",")
                        np.save(args.outputFilename + "_bruteForceCurrentDesign.npy", var)

                if cost <= costTolerance:
                    if args.outputFilename != None:
                        np.savetxt(args.outputFilename + "_bruteForceCurrentCosts.txt", np.array(costs), delimiter=",")
                        np.save(args.outputFilename + "_bruteForceCurrentDesign.npy", var)

                    return var, cost
                
    if args.outputFilename != None:
        np.savetxt(args.outputFilename + "_bruteForceCurrentCosts.txt", np.array(costs), delimiter=",")
        np.save(args.outputFilename + "_bruteForceCurrentDesign.npy", var)
    return var, cost
        

if __name__ == "__main__":

    AU = 1.496e11 # Astronomical Unit (m)

    dT_default = 5 #default time step = 5 days
    randomSeed_default = 0 #default randomseed is 0
    # bruteForce_default = 1 #default: don't run bruteForce
    numNetworkSails_default = 10 #default: 10 sails in the network
    runOptimization_default = 1 #default: run optimization. if 0, just run with given pre-optimized network design
    NEOname_default = 'Vesta' #default: target NEO object is Vesta

    parser = argparse.ArgumentParser(
                        prog='LightSail_GA.py',
                        description='Genetic algorithm for single trajectory optimization')
    parser.add_argument('--inputFilename', nargs='+')       # filename for saving final optimized var -- if no filename given, no save. give without file extension
    parser.add_argument('--outputFilename')       # filename for saving final optimized var -- if no filename given, no save. give without file extension
    parser.add_argument('-rs', '--randomSeed')    # random seed value
    parser.add_argument('-dT', '--dT')            # time step size (days)
    parser.add_argument('--numNetworkSails')      # number of sails in the network
    parser.add_argument('--runOptimization')      # run optimization or just run network with pre-optimized network design
    parser.add_argument('--preOptimizedNetworkDesign')      # number of sails in the network
    parser.add_argument('--NEOname')              # Name of target object for simulation. options: 'NEO', '16 Psyche', 'Vesta', 'Eunomia', 'Ceres'
    # parser.add_argument('-b', '--bruteforce')     # bruteforce optimization option -- run if 1

    args = parser.parse_args()

    dT = float(args.dT) if args.dT!=None else dT_default
    seed = args.randomSeed if args.randomSeed!=None else randomSeed_default
    numNetworkSails = int(args.numNetworkSails) if args.numNetworkSails!=None else numNetworkSails_default
    runOptimization = int(args.runOptimization) if args.runOptimization!=None else runOptimization_default
    NEOname = args.NEOname if args.NEOname!=None else NEOname_default
    # bruteForce = int(args.bruteforce) if args.bruteforce!=None else bruteForce_default
    if args.inputFilename is None:
        print("There is no input filename")
        sys.exit()
    else:
        pathfiles = args.inputFilename

    np.random.seed(int(seed))

    # Import pre-optimized paths for sails going to specific NEOs:
    optimizedPathPos = []
    indexAndTimeofNEOVisit = [] #list with entries in the format: [index of object that visited NEO, earliest time of visit to NEO]
    distForNEOVisit = 100000e3/AU #distance that counts as a visit to a NEO

    i = 1
    for pathfile in pathfiles:
        df = pd.read_csv(pathfile)
        path = np.column_stack((df['sailPosX'], df['sailPosY'], df['sailPosZ'], df['sailActive']))
        optimizedPathPos.append(path)
        visitIndex = np.where(df['distanceToNEO'] < distForNEOVisit)[0]
        indexAndTimeofNEOVisit.append([i,df['Time'].values[visitIndex][0]])
        i+=1

    earthPos = np.column_stack((df['earthPosX'], df['earthPosY'], df['earthPosZ'], np.ones_like(df['earthPosY']))) #x, y, z position, then "active" status

    optimizedSailTakeOffTime = np.searchsorted(df['sailActive'],1)*dT

    T = 365*5 # Maximum simulation time (days) 

    # ALL CELESTIAL BODY DATA BELOW IS FOR JULY 12, 2024

    # Earth's Orbital Elements
    print("Calculating Earth trajectory..")
    earthPos_ = np.empty((0, 3))
    earthVel = np.empty((0, 3))
    Earth = orbitingObject(1.496430050492096e11/AU, 0.01644732672533337, np.radians(0.002948905108822335), np.radians(250.3338397589344), np.radians(211.4594045093653),  np.radians(188.288909341482), 5.97219e24, 'Earth', earthPos_, earthVel)
    Earth = calcCelestialTraj(Earth, dT, T)

    # NEO 2016 VO1 Orbital Elements
    if NEOname == 'NEO':
        print("Calculating NEO trajectory..")
        NEOPos = np.empty((0, 3))
        NEOVel = np.empty((0, 3))
        NEO = orbitingObject(146992149*1000/AU, 0.42869653, np.radians(1.61582328), np.radians(131.95199360416), np.radians(129.74129871507), np.radians(5.517059440296369e1), 0, 'NEO', NEOPos, NEOVel) 
        NEO = calcCelestialTraj(NEO, dT, T)
        targetObject = NEO

    # # 16 Psyche
    if NEOname == '16 Psyche':
        pschePos = np.empty((0, 3))
        psycheVel = np.empty((0, 3))
        Psyche = orbitingObject(4.372304796380328e11 / AU, 0.134153724002426, np.radians(3.092472666018101), np.radians(229.5568276039235), np.radians(150.019510189129), np.radians(302.7603308460203), 1.53, '16 Psyche', pschePos, psycheVel)
        Psyche = calcCelestialTraj(Psyche, dT, T)
        targetObject = Psyche

    # # Vesta 
    if NEOname == 'Vesta':
        vestaPos = np.empty((0, 3))
        vestaVel = np.empty((0, 3))
        Vesta = orbitingObject(3.531823140411854E+11 / AU, 8.996689130349783E-02, np.radians(7.14181494423702), np.radians(1.516915946723933E+02), np.radians(1.037050935401676E+02), np.radians(2.516482952688035E+02), 17.28824, 'Vesta', vestaPos, vestaVel)
        Vesta = calcCelestialTraj(Vesta, dT, T)
        targetObject = Vesta

    # Eunomia = orbitingObject(A, EC, IN, w, OM, MA, mass, name)
    if NEOname == 'Eunomia':
        eunomiaPos = np.empty((0, 3))
        eunomiaVel = np.empty((0, 3))
        Eunomia = orbitingObject(3.954262791032822E+11 / AU, 1.873532866281441E-01, np.radians(1.175476877128721E+01), np.radians(9.877585450769357E+01), np.radians(2.928995130001719E+02), np.radians(3.594995210917758E+02), 17.28824, 'Eunomia', eunomiaPos, eunomiaVel)
        Eunomia = calcCelestialTraj(Eunomia, dT, T)
        targetObject = Eunomia

    # Ceres = orbitingObject(A, EC, IN, w, OM, MA, mass, name)
    if NEOname == 'Ceres':
        ceresPos = np.empty((0, 3))
        ceresVel = np.empty((0, 3))
        Ceres = orbitingObject(4.139129887889629E+11 / AU, 7.910717043218352E-02, np.radians(1.058782499528511E+01), np.radians(7.331580895116618E+01), np.radians(8.025383722598423E+01), np.radians(1.250473243162762E+02), 17.28824, 'Ceres', ceresPos, ceresVel)
        Ceres = calcCelestialTraj(Ceres, dT, T)
        targetObject = Ceres

    # Bennu = orbitingObject(A, EC, IN, w, OM, MA, mass, name)
    if NEOname == 'Bennu':
        print("Calculating Bennu trajectory...")
        bennuPos = np.empty((0, 3))
        bennuVel = np.empty((0, 3))
        Bennu = orbitingObject(1.684403508572353E+11 / AU, 2.037483028559170E-01, np.radians(6.032932274441114E+00), np.radians(6.637388139157433E+01), np.radians(1.981305199928344E+00), np.radians(2.175036361198920E+02), 7.329E10, 'Bennu', bennuPos, bennuVel) #mass 7.329E10 from: https://www.nature.com/articles/s41550-019-0721-3
        Bennu = calcCelestialTraj(Bennu, dT, T)
        targetObject = Bennu

    NEOPos = targetObject.positions

    # Desired designs and weights
    w1 = 2 #end goal of getting info from NEO visits back to Earth -- should be >=1
    w2 = 10 #number of connections
    w3 = 10000000 #min distance out of all network sails to earth
    w4 = 10 #min distance out of all network sails to the sail going to NEO
    w5 = 1000 #sum of min dist between network sails and every other network sail
    w6 = 100000 #sum of max dist between network sails and every other network sail
    w7 = 10000000 #penalty for flying too close to the sun

    w = [w1, w2, w3, w4, w5, w6, w7]

    TOLNEO=0.1
    TOLEarth=0.1

    solver = lightSailNetworkSolver()

    if runOptimization == 1:

        #Define genetic algorithm parameters
        S=30
        P=10
        K=10
        G=100
        TOL=0.175

        #Design Variables min and max
        NumSeg= 30
        max_variation=0.5
        degree_min=1
        # degree_max=4
        degree_max=1
        # time_min=dT #launch time
        # time_max=365
        time_min=optimizedSailTakeOffTime #launch time -- 
        time_max=optimizedSailTakeOffTime
        vel_min=30*1000/AU # (AU/s)
        vel_max=30*1000/AU # (AU/s)
        setInitDir_min = 0
        setInitDir_max = 1
        timeSeg_min=np.full(NumSeg,1)
        timeSeg_max=np.full(NumSeg,70)
        coneAngle_min = np.full(NumSeg, -70*np.pi/180) #See BLISS paper -- keep cone angle between -70 to 70 deg. to prevent loss of control authority
        coneAngle_max = np.full(NumSeg, 70*np.pi/180)
        clockAngle_min = np.full(NumSeg, -180*np.pi/180)
        clockAngle_max = np.full(NumSeg, 180*np.pi/180)

        #Number of design variables
        dv=3+NumSeg*3

        #Upper and lower bounds for each variable
        lb = np.concatenate([[degree_min, time_min, vel_min],timeSeg_min,coneAngle_min,clockAngle_min])
        ub = np.concatenate([[degree_max, time_max, vel_max],timeSeg_max,coneAngle_max,clockAngle_max])

        print("lower bound shape", lb.shape, ub)

        start_time = time.time()

        Lambda, Pi, Orig, meanPi, minPi, meanParents, cost = GeneticAlgorithm(S,P,K,TOL,G,dv,lb,ub,w,solver.calcNetworkCost,max_variation, NumSeg, numNetworkSails, optimizedPathPos, earthPos, indexAndTimeofNEOVisit, dT, T, render = 0)

        solver.runNetworkSim(Lambda[0,:,:], earthPos, optimizedPathPos, dT, T, indexAndTimeofNEOVisit, prevEarthAndSailPos = None, toRerun = None, render = 1, trackNEO = 1, NEOPos = NEOPos, w = w, output_filename = args.outputFilename)
        
        var, cost = bruteForceOptimization(Lambda[0,:,:], w, dT, T, optimizedPathPos, earthPos, indexAndTimeofNEOVisit, lb, ub, max_iter=20, tolerance=150, m=10, printStatements = 1, costTolerance = TOL)

        solver.runNetworkSim(var, earthPos, optimizedPathPos, dT, T, indexAndTimeofNEOVisit, prevEarthAndSailPos = None, toRerun = None, render = 1, w = w, output_filename = args.outputFilename)

        end_time = time.time()
        total_run_time = end_time - start_time
        print(f"Total run time: {total_run_time} seconds")

    elif runOptimization==0:
        if args.preOptimizedNetworkDesign is None:
            print("No pre-optimized network design given")
        else:
            print("NEOPos", NEOPos)
            var = np.load(args.preOptimizedNetworkDesign)
            solver.runNetworkSim(var, earthPos, optimizedPathPos, dT, T, indexAndTimeofNEOVisit, prevEarthAndSailPos = None, toRerun = None, render = 1, w = w, output_filename = args.outputFilename, trackNEO = 1, NEOPos = NEOPos)

