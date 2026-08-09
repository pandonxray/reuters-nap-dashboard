$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Port = 8522
$Url = "http://localhost:$Port"
$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$FallbackPython = "D:\Anacoda\python.exe"
$Python = $VenvPython
$OutputDir = Join-Path $ProjectDir "outputs"
$StdoutLog = Join-Path $OutputDir "nap_streamlit.out.log"
$StderrLog = Join-Path $OutputDir "nap_streamlit.err.log"
$LauncherLog = Join-Path $OutputDir "nap_launcher.log"

function Write-LauncherLog {
    param([string]$Message)
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $LauncherLog -Value "[$stamp] $Message" -Encoding UTF8
}

function Show-LauncherMessage {
    param([string]$Message)
    try {
        $shell = New-Object -ComObject WScript.Shell
        [void]$shell.Popup($Message, 8, "Reuters NAP Dashboard", 48)
    } catch {
        Write-LauncherLog $Message
    }
}

function Test-NapPort {
    try {
        return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    } catch {
        return $false
    }
}

function Get-NapPortProcess {
    try {
        $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $connection) {
            return $null
        }
        return Get-CimInstance Win32_Process -Filter "ProcessId = $($connection.OwningProcess)" -ErrorAction SilentlyContinue
    } catch {
        return $null
    }
}

try {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

    $existingProcess = Get-NapPortProcess
    if ($existingProcess) {
        $expectedPrefix = (Join-Path $ProjectDir ".venv").ToLowerInvariant()
        $existingExecutable = [string]$existingProcess.ExecutablePath
        $existingCommand = [string]$existingProcess.CommandLine
        $isNapDashboard = $existingCommand -like "*nap_dashboard.py*"
        $usesProjectEnvironment = $existingExecutable.ToLowerInvariant().StartsWith($expectedPrefix) -or $existingCommand -like "*-m streamlit*"
        if ($isNapDashboard -and -not $usesProjectEnvironment) {
            $message = "Port $Port is occupied by an older NAP Dashboard running outside the project environment.`nClose that dashboard process, then start again."
            Write-LauncherLog "$message Executable: $existingExecutable"
            Show-LauncherMessage $message
            exit 1
        }
        Write-LauncherLog "Port $Port is already listening with the expected runtime; opening $Url."
        Start-Process $Url
        exit 0
    }

    if (-not (Test-Path -LiteralPath $Python)) {
        $Python = $FallbackPython
    }

    if (-not (Test-Path -LiteralPath $Python)) {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd) {
            $Python = $cmd.Source
        } else {
            Show-LauncherMessage "Python was not found: $VenvPython or $FallbackPython`nPlease check the project environment."
            exit 1
        }
    }

    Write-LauncherLog "Starting Streamlit from $ProjectDir on port $Port using $Python -m streamlit."
    Start-Process `
        -FilePath $Python `
        -ArgumentList @("-m", "streamlit", "run", "src\nap_dashboard.py", "--server.port", "$Port", "--server.headless", "true", "--browser.gatherUsageStats", "false", "--server.fileWatcherType", "none", "--server.runOnSave", "false") `
        -WorkingDirectory $ProjectDir `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog `
        -WindowStyle Hidden

    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        if (Test-NapPort) {
            Start-Process $Url
            exit 0
        }
    }

    Show-LauncherMessage "Dashboard startup timed out. Check the log:`n$StderrLog"
    exit 1
} catch {
    Write-LauncherLog ("Launcher failed: " + $_.Exception.Message)
    Show-LauncherMessage ("Startup failed: " + $_.Exception.Message + "`nLog: " + $LauncherLog)
    exit 1
}
