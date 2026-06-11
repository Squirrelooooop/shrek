This pipeline uses caiman and cellpose to analyze calcium imaging of hek cells and output raw F of each cell over time.

There are two imaging systems currently used in the lab: 
1) Upright system using HCimaging software which first output video format is dcimg.
2) Inverted system using Micro-Manager2.0 which first output format is ome.tiff.

Setup: 
1. Download Fiji ImageJ: https://imagej.net/software/fiji/downloads
2. Create a Github account and download Github Desktop. Fork this to your repository and clone it to your local folder.
   <img width="709" height="180" alt="Screenshot 2026-06-05 at 5 26 39 PM" src="https://github.com/user-attachments/assets/194d98f4-44ae-4e50-ac2c-8359fbcc0c8f" />
   <img width="494" height="497" alt="Screenshot 2026-06-05 at 5 27 46 PM" src="https://github.com/user-attachments/assets/ba271b60-b36c-45e0-b779-11dc29470fbb" />
3. Download Visual Studio Code (or any IDE you like to use)
4. Download and setup miniforge: https://github.com/conda-forge/miniforge
5. Create conda environment in Terminal for running [cellpose](https://github.com/MouseLand/cellpose) and [caiman](https://github.com/flatironinstitute/caiman) respectively
   For cellpose:
   ```
   conda create -n cellpose_py310 python=3.10
   conda activate cellpose_py310
   pip install cellpose
   conda install pandas
   conda deactivate
   ```
   For caiman:
   ```
   mamba create -n caiman caiman
   ```
   
As current workflow of the imaging analysis: convert the video to multipage tiff --> downsample (with rolling average) --> motion correction by caiman --> cell segmentation, assigning cell id and tag+ cells by cellpose --> extract raw F value for each cell (with background subtraction) by MATLAB.
1. Converting to multipage tiffs with extension ".tif" or ".tiff"
   1) Upright system
      
      a. first do batch conversion in the HCimaging from dcimg to cxd
      b. Use Fiji ImageJ to convert from cxd to tif: Open Fiji > Process > Batch > Convert... > <img width="494" height="273" alt="Screenshot 2026-06-05 at 5 21 34 PM" src="https://github.com/user-attachments/assets/f89076f3-f7e9-4e62-8864-57c8ec966c55" />
   2) Inverted system: do this step in workflow.py:
      Open workflow.py with VS Code.

      First you need to switch to cellpose_py310 environment: From the top search bar choose "Show and Run Commands >", then "Python: Select Interpreter", and then choose cellpose_py310
      
      Keep the first line uncommented, uncomment and change the input_dir as your data folder path ```cp.ometotiff(input_dir = "/Users/The_folder_where_your_ome_tiff_is_stored/")``` and comment all other lines.
      
      If you are using windows system and the folder path you copy has ```\```, replace this line with ```cp.ometotiff(input_dir = r"\Users\The_folder_where_your_ome_tiff_is_stored\")```.

      Then hit run on the top right corner.

2. Downsample

   Open workflow.py with VS Code.

   First you need to switch to cellpose_py310 environment: From the top search bar choose "Show and Run Commands >", then "Python: Select Interpreter", and then choose cellpose_py310

   Keep the first line uncommented (dont change this line), uncomment ```cp.downsample(input_dir = "/Users/Your_tif_folder_path/", msPerFrame=60, ds_factor = 10)```. Remember to change the msPerFrame to your actual calcium recording frame interval.

4. Do the motion correction in sunny_caiman_motion_correction.py and remember to change the conda env to caiman (Optional)

   Open sunny_caiman_motion_correction.py with VS Code, switch to caiman environment: From the top search bar choose "Show and Run Commands >", then "Python: Select Interpreter", and then choose caiman.

   Change the input_dir to your downsampled tif and hit run on the top right corner of the VS Code window

   Close the python file after it is done. Check each tif and delete the ones with incorrect motion correction

6. Cellpose: go back to workflow.py > comment every lines of code except for the first and last line. Change the input_dir to your folder path storing motion corrected tifs (have "_mc" in the filename). If you have snap tif as tag+ reference, add the folder path of where you store those tiffs to snap_dir = "". Then hit run.

7. MATLAB: run callum_optimize_hek.m with MATLAB. Change the input directory ```folder = '';``` to your cellpose output folder. Pay attention to the sector %% --- Time vector ---; you may need to change dt = time interval between frames of your downsampled tif.

8. Actual analysis of the calcium data. Depending on which coding language you use, most prefer outputs as csv files. This step convert all output files needed for plot to csv.

  (1) CaIData: open savecsv.m, change the input directory to to where you store CalData output files ```matFiles = dir(fullfile('foldwe_path_of_your_CalData_files', '*.mat'));```

     Pay attention to ```arrayToWrite = [cellIDs; data.F];```. This saves the raw F value. If you want the dF/F calculated in callum_optimize_hek.m replace this to ```arrayToWrite = [cellIDs; data.DFoverF]```, but I recommend to start with raw F values.

  (2) Timestamp. If you use TimestampSunny.m during your recording session to note down the exact timing of camera-On and ultrasound-On, then you can use this to output the timing of the ultrasound and match this with your calcium traces. Change the directory of where you store the timestamp matlab files and where want to put the csv files:
     ```folder = 'input_folder_path'; out_folder = 'output_folder_must_be_created_before_running_this_script';```

9. Run the rest of the analysis in your familiar coding languages. Good Luck! If you r using R/Rstudio, check the server/ARPA/CalciumImaging folders to see if you can use scripts from previous experiments, especially if you r following the same imaging protocol. 
   



