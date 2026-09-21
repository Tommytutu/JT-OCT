$ErrorActionPreference = "Stop"
$D3Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$D3Source = Join-Path $D3Root "native\interval_oracle.cpp"
$D3OutputDir = Join-Path $D3Root "jt_oct\_native"
$D3Output = Join-Path $D3OutputDir "interval_oracle.dll"
New-Item -ItemType Directory -Path $D3OutputDir -Force | Out-Null
$D3Object = Join-Path $D3OutputDir "interval_oracle.obj"
$D3Import = Join-Path $D3OutputDir "interval_oracle.lib"
$D3VsWhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
$D3Install = & $D3VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $D3Install) { throw "MSVC C++ tools were not found" }
$D3VcVars = Join-Path $D3Install "VC\Auxiliary\Build\vcvars64.bat"
$D3Command = '"' + $D3VcVars + '" >nul && cl /nologo /O2 /openmp /EHsc /std:c++17 /LD /Fo"' + $D3Object + '" "' + $D3Source + '" /link /OUT:"' + $D3Output + '" /IMPLIB:"' + $D3Import + '"'
cmd.exe /d /s /c $D3Command
if ($LASTEXITCODE -ne 0) { throw "Interval oracle build failed" }
Write-Output $D3Output
