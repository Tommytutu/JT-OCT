$ErrorActionPreference = 'Stop'
$PrepareRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PrepareSource = Join-Path $PrepareRoot 'native\d3_prepare.cpp'
$PrepareOutput = Join-Path $PrepareRoot 'jt_oct\_native\d3_prepare.dll'
$PrepareObject = Join-Path $PrepareRoot 'jt_oct\_native\d3_prepare.obj'
$PrepareImport = Join-Path $PrepareRoot 'jt_oct\_native\d3_prepare.lib'
$PrepareVsWhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$PrepareInstall = & $PrepareVsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $PrepareInstall) { throw 'MSVC C++ tools were not found' }
$PrepareVcVars = Join-Path $PrepareInstall 'VC\Auxiliary\Build\vcvars64.bat'
$PrepareCommand = '"' + $PrepareVcVars + '" >nul && cl /nologo /O2 /EHsc /std:c++17 /LD /Fo"' + $PrepareObject + '" "' + $PrepareSource + '" /link /OUT:"' + $PrepareOutput + '" /IMPLIB:"' + $PrepareImport + '"'
cmd.exe /d /s /c $PrepareCommand
if ($LASTEXITCODE -ne 0) { throw 'D3 native preparation build failed' }
Write-Output $PrepareOutput
