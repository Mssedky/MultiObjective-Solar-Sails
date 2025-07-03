# SolarSail

## To create and activate conda environment:
```
conda create --name sail --file requirements.txt
conda activate sail
```

## To replicate Case 3 figures:
```
python LightSailNetworkSolver.py --inputFilename "BennuTraj.csv" --dT 5 --numNetworkSails 10 --preOptimizedNetworkDesign "BennuNetworkOptimal_bruteForceCurrentDesign.npy" --runOptimization 0 --outputFilename "BennuNetworkOptimal" --NEOname "Bennu" 
```

## To generate your own networks:

First, generate the sail path directly to the NEO 101955 Bennu (or, choose "Eunomia", "Vesta", "Ceres", "16 Psyche"), with 0 weight on returning to Earth (same process as Case 1, but only rendezvous, no return). This should output pre-optimized-sail-var.txt:

```
python LightSail_GA.py --bruteforce 1 --dT 5 --T 1825 --randomseed 2 --filename "pre-optimized-sail-var.txt" --experiments 0 --solarSysExit 0 --NEOname "Bennu"
```

Run the optimized sail path from above to generate trajectory data. This should output pre-optimized-sail-traj.dat and pre-optimized-sail-traj.csv:

```
python Test.py --inputfilename "pre-optimized-sail-var.txt" --savetraj 1 --outputfilename "pre-optimized-sail-traj" --dT 5 --NEOname "Bennu" --T 1825
```

Lastly, run the network solver to optimize for data return to Earth. This should output BennuNetworkOptimal_currentCosts.txt, BennuNetworkOptimal_currentDesigns.npy, BennuNetworkOptimal_bruteForceCurrentCosts.txt, BennuNetworkOptimal_bruteForceCurrentDesign.npy:
```
python LightSailNetworkSolver.py --inputFilename "pre-optimized-sail-traj.csv" --dT 5 --numNetworkSails 10 --randomSeed 3 --outputFilename "BennuNetworkOptimal"
```

