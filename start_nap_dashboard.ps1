$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Port = 8522
$Url = "http://localhost:$Port"
$Streamlit = "D:\Anacoda\Scripts\streamlit.exe"
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

try {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

    if (Test-NapPort) {
        Write-LauncherLog "Port $Port is already listening; opening $Url."
        Start-Process $Url
        exit 0
    }

    if (-not (Test-Path -LiteralPath $Streamlit)) {
        $cmd = Get-Command streamlit -ErrorAction SilentlyContinue
        if ($cmd) {
            $Streamlit = $cmd.Source
        } else {
            Show-LauncherMessage "Streamlit was not found: $Streamlit`nPlease check the Anaconda/Streamlit installation."
            exit 1
        }
    }

    Write-LauncherLog "Starting Streamlit from $ProjectDir on port $Port."
    Start-Process `
        -FilePath $Streamlit `
        -ArgumentList @("run", "src\nap_dashboard.py", "--server.port", "$Port", "--server.headless", "true", "--browser.gatherUsageStats", "false") `
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
