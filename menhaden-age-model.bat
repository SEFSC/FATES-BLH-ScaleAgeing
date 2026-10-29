@echo off
echo Checking system requirements...

:: Search for Python 3.14 to match requirements.txt
set "PYTHON_EXE="

:: Method A: Try the Windows Python Launcher
py -3.14 --version >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    set "PYTHON_EXE=py -3.14"
    GOTO :VersionOK
)

:: Method B: Check the standard user-level installation directories
IF EXIST "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" (
    set "PYTHON_EXE="%LOCALAPPDATA%\Programs\Python\Python314\python.exe""
    GOTO :VersionOK
)

:: Method C: Check the pythoncore-3.14-64 directory 
IF EXIST "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" (
    set "PYTHON_EXE="%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe""
    GOTO :VersionOK
)

:: If we reach here, 3.14 was not found on the system
GOTO :InstallPython

:InstallPython
echo.
echo ==========================================================
echo ERROR: Python 3.14 was not found on your system.
echo ==========================================================
echo This application requires Python 3.14 to run the modern
echo machine learning dependencies.
echo.
echo Please install Python 3.14 using the website that is 
echo about to open.
echo.
echo NOTE: If this is the first Python installation on your
echo workstation, checking the "Add Python to PATH" box during
echo installation is strongly recommended.
echo ==========================================================
echo.
echo Press any key to open the download page and exit...
pause >nul
start https://www.python.org/downloads/
exit /b

:VersionOK
:: Set the working directory to the location of this .bat file
cd /d "%~dp0"

:: Check if the '.venv' folder already exists
IF NOT EXIST ".venv\" (
    echo.
    echo Compatible Python version found.
    echo First-time setup: Creating Python environment...
    
    :: Use the specifically located Python to create the environment
    %PYTHON_EXE% -m venv .venv
    
    :: Halt immediately if the virtual environment fails to create
    if errorlevel 1 (
        echo.
        echo ==========================================================
        echo ERROR: Failed to create the virtual environment!
        echo ==========================================================
        pause
        exit /b
    )
    
    echo Activating environment and downloading dependencies...
    echo ^(This may take several minutes. Do not close this window.^)
    call .venv\Scripts\activate.bat
    
    :: Once activated, upgrade pip
    python -m pip install --upgrade pip >nul
    
    :: Install from requirements.txt
    pip install -r requirements.txt
    
    :: ERROR CATCHER: Dynamic check that halts if pip install fails
    if errorlevel 1 (
        echo.
        echo ==========================================================
        echo ERROR: Failed to download and install dependencies!
        echo ==========================================================
        echo Removing broken environment...
        call .venv\Scripts\deactivate.bat >nul 2>&1
        cd /d "%~dp0"
        rmdir /s /q .venv
        echo.
        echo Please check your internet connection and try running this script again.
        pause
        exit /b
    )
) ELSE (
    echo Python Environment found! Activating...
    call .venv\Scripts\activate.bat
)

:: Prompt the user to select the Model Type via a custom GUI dialogue box
echo.
echo Launching model selection window...

:: Create a temporary PowerShell script to generate the custom Form
echo Add-Type -AssemblyName System.Windows.Forms > "%TEMP%\ModelPrompt.ps1"
echo $form = New-Object System.Windows.Forms.Form >> "%TEMP%\ModelPrompt.ps1"
echo $form.Text = 'Menhaden Ageing Model Launcher' >> "%TEMP%\ModelPrompt.ps1"
echo $form.Size = New-Object System.Drawing.Size(350,230) >> "%TEMP%\ModelPrompt.ps1"
echo $form.StartPosition = 'CenterScreen' >> "%TEMP%\ModelPrompt.ps1"
echo $form.FormBorderStyle = 'FixedDialog' >> "%TEMP%\ModelPrompt.ps1"
echo $form.MaximizeBox = $false >> "%TEMP%\ModelPrompt.ps1"

echo $label = New-Object System.Windows.Forms.Label >> "%TEMP%\ModelPrompt.ps1"
echo $label.Text = 'What would you like to do?' >> "%TEMP%\ModelPrompt.ps1"
echo $label.AutoSize = $true >> "%TEMP%\ModelPrompt.ps1"
echo $label.Location = New-Object System.Drawing.Point(20,20) >> "%TEMP%\ModelPrompt.ps1"
echo $form.Controls.Add($label) >> "%TEMP%\ModelPrompt.ps1"

echo $btnProcess = New-Object System.Windows.Forms.Button >> "%TEMP%\ModelPrompt.ps1"
echo $btnProcess.Text = 'Process raw images (crop, pad, normalize, etc.)' >> "%TEMP%\ModelPrompt.ps1"
echo $btnProcess.Location = New-Object System.Drawing.Point(20,50) >> "%TEMP%\ModelPrompt.ps1"
echo $btnProcess.Size = New-Object System.Drawing.Size(290,30) >> "%TEMP%\ModelPrompt.ps1"
echo $btnProcess.Add_Click({$form.Tag = 'process'; $form.Close()}) >> "%TEMP%\ModelPrompt.ps1"
echo $form.Controls.Add($btnProcess) >> "%TEMP%\ModelPrompt.ps1"

echo $btnImage = New-Object System.Windows.Forms.Button >> "%TEMP%\ModelPrompt.ps1"
echo $btnImage.Text = 'Predict ages using images only' >> "%TEMP%\ModelPrompt.ps1"
echo $btnImage.Location = New-Object System.Drawing.Point(20,90) >> "%TEMP%\ModelPrompt.ps1"
echo $btnImage.Size = New-Object System.Drawing.Size(290,30) >> "%TEMP%\ModelPrompt.ps1"
echo $btnImage.Add_Click({$form.Tag = 'images'; $form.Close()}) >> "%TEMP%\ModelPrompt.ps1"
echo $form.Controls.Add($btnImage) >> "%TEMP%\ModelPrompt.ps1"

echo $btnMulti = New-Object System.Windows.Forms.Button >> "%TEMP%\ModelPrompt.ps1"
echo $btnMulti.Text = 'Predict ages using images and metadata' >> "%TEMP%\ModelPrompt.ps1"
echo $btnMulti.Location = New-Object System.Drawing.Point(20,130) >> "%TEMP%\ModelPrompt.ps1"
echo $btnMulti.Size = New-Object System.Drawing.Size(290,30) >> "%TEMP%\ModelPrompt.ps1"
echo $btnMulti.Add_Click({$form.Tag = 'metadata'; $form.Close()}) >> "%TEMP%\ModelPrompt.ps1"
echo $form.Controls.Add($btnMulti) >> "%TEMP%\ModelPrompt.ps1"

echo $form.ShowDialog() ^| Out-Null >> "%TEMP%\ModelPrompt.ps1"
echo Write-Output $form.Tag >> "%TEMP%\ModelPrompt.ps1"

:: Execute the temporary script and capture the choice
for /f "usebackq tokens=*" %%I in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%TEMP%\ModelPrompt.ps1"`) do set "MODEL_CHOICE=%%I"

:: Clean up the temporary file
del "%TEMP%\ModelPrompt.ps1"

:: Assign the correct script based on the choice, or exit if they closed the window
IF "%MODEL_CHOICE%"=="process" (
    set "SCRIPT_NAME=process-images.py"
    echo Selected Action: Process raw images
) ELSE IF "%MODEL_CHOICE%"=="images" (
    set "SCRIPT_NAME=predict-ages-images.py"
    echo Selected Action: Predict ages using images only
) ELSE IF "%MODEL_CHOICE%"=="metadata" (
    set "SCRIPT_NAME=predict-ages-multimodal.py"
    echo Selected Action: Predict ages using images and metadata
) ELSE (
    echo.
    echo No action was selected. The program will now exit.
    pause
    exit /b
)

:: Prompt the user to select the configurations.yml file
echo.
echo Please select your configuration file from the pop-up window...

:: Define the PowerShell command to open the File Dialog
set "psCommand=Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.OpenFileDialog; $f.Filter = 'YAML Files (*.yml;*.yaml)|*.yml;*.yaml|All Files (*.*)|*.*'; $f.Title = 'Select Configuration File'; $f.InitialDirectory = '%CD%'; if ($f.ShowDialog() -eq 'OK') { Write-Output $f.FileName }"

:: Execute the PowerShell command and capture the output into the CONFIG_FILE variable
for /f "usebackq tokens=*" %%I in (`powershell -NoProfile -Command "%psCommand%"`) do set "CONFIG_FILE=%%I"

:: Check if the user cancelled the prompt or didn't select a file
IF "%CONFIG_FILE%"=="" (
    echo.
    echo No configuration file was selected. The program will now exit.
    pause
    exit /b
)

echo Selected configuration: "%CONFIG_FILE%"

:: Run your actual model script based on the choices
echo.
echo Starting the model...
python -u scripts\%SCRIPT_NAME% --config_path "%CONFIG_FILE%"

:: Keep the command prompt open after the script finishes or errors out
echo.
pause
