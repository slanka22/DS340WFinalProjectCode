# **DreamerV2 as an improvement for Traffic Signal Management**
## *By: Srikar Lanka and Kunjan Patel*
### Adapting **DreamerV2** by esteveste with **"A Holistic Framework Towards Vision-based Traffic Signal Control with Microscopic Simulation"** by Pan He, Quanyi Li, Xiaoyong (Brian) Yuan, and Bolei Zhou
---
**Note:** Current install directions are for Windows only (tested on Windows 11). Future directions may be released for MacOS or verified for other installations of Windows. 
<br></br>
# Prior to Run To-Dos
## Part 1: Python Installation Check
**Note:** This project requires Python 3.10. Please ensure you have this installed. 

## Part 2: Downloading SUMO
### Visit Website to Download Link:

 1. Visit: [SUMO Official Website](https://sumo.dlr.de/docs/Downloads.php)
 2. Download Installer
 3. Run Installer
 4. Ensure to Check Add to Path Checkmark in Installer
	 1. There is a chance this may not work, in-which you will need to add it to your PATH manually. This can be done by verifying where your installation utilizing the steps below, and then also adding said location to your PATH. 
 5. Finish Installation

### Configuration Step:

 1. Locate where SUMO was installed to
	 1. Example: Shortcut to Sumo Points to "C:\Program Files (x86)\Eclipse\Sumo\"
 2. Open PowerShell
 3. Run `setx SUMO_HOME "<FILE PATH>"`
	 1. Example: setx SUMO_HOME "C:\Program Files (x86)\Eclipse\Sumo"
 4. Restart PowerShell

 
 ## Part 3: Open ImplementationPlatform.py (if using IDE)
 
 ## Part 4: Install Packages
 ### Run Following Commands (In-Order)
 

 1. ` python.exe -m pip install --upgrade pip   `
 2. `pip install numpy`
 3. `pip install pandas`
 4. `pip install torch`
 5. `pip install PyYAML`
 6. `pip install opencv-python`
 7. `pip install gym`
 8. `pip install elements`
 9. `pip install gymnasium`
 10. `pip install pettingzoo`
 11. `pip install metadrive-simulator==0.4.3` 

 ## Part 5: Run Code File
 1. Ensure you are running the project from the project root folder if not using a IDE.
  
 **Note:** To ensure universal compatibility for the GitHub release, PyTorch for the model is configured to run in CPU-mode. This will take time, but do not worry, more then likely the case is that your code is running. You are welcome to change the configuration to cuda or xpu based on your device specifications, though this readme will not directly provide instructions to do so at this time. 
