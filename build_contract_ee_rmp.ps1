param([string]$GurobiRoot = 'C:\gurobi1300\win64')
$ErrorActionPreference = 'Stop'
$EeRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EeOutput = Join-Path $EeRoot 'jt_oct\_native'
$EeVsWhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$EeInstall = & $EeVsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $EeInstall) { throw 'MSVC C++ tools were not found' }
if (-not (Test-Path -LiteralPath (Join-Path $GurobiRoot 'include\gurobi_c++.h'))) { throw 'Gurobi C++ SDK was not found' }
New-Item -ItemType Directory -Path $EeOutput -Force | Out-Null
$EeVcVars = Join-Path $EeInstall 'VC\Auxiliary\Build\vcvars64.bat'
$EeCommand = '"' + $EeVcVars + '" >nul && cl /nologo /O2 /MD /EHsc /std:c++17 /LD /I"' + $GurobiRoot + '\include" /Fo"' + $EeOutput + '\contract_ee_rmp.obj" "' + $EeRoot + '\native\contract_ee_rmp.cpp" /link /LIBPATH:"' + $GurobiRoot + '\lib" gurobi_c++md2017.lib gurobi130.lib /OUT:"' + $EeOutput + '\contract_ee_rmp.dll" /IMPLIB:"' + $EeOutput + '\contract_ee_rmp.lib"'
cmd.exe /d /s /c $EeCommand
if ($LASTEXITCODE -ne 0) { throw 'Endpoint-elimination C++ RMP build failed' }
