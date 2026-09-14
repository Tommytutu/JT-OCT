$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Source = Join-Path $ProjectRoot "native\jt_message.cpp"
$OutputDir = Join-Path $ProjectRoot "jt_oct\_native"
$Output = Join-Path $OutputDir "jt_message.dll"
$Object = Join-Path $OutputDir "jt_message.obj"
$ImportLibrary = Join-Path $OutputDir "jt_message.lib"
$VsWhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
$Install = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $Install) { throw "MSVC C++ tools were not found" }
$VcVars = Join-Path $Install "VC\Auxiliary\Build\vcvars64.bat"
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$Command = '"' + $VcVars + '" >nul && cl /nologo /O2 /openmp /EHsc /std:c++17 /LD /Fo"' + $Object + '" "' + $Source + '" /link /OUT:"' + $Output + '" /IMPLIB:"' + $ImportLibrary + '"'
cmd.exe /d /s /c $Command
if ($LASTEXITCODE -ne 0) { throw "Direct message-passing build failed" }
Write-Output $Output
