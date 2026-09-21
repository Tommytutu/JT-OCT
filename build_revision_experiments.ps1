param([string]$GurobiRoot = 'C:\gurobi1300\win64')
$ErrorActionPreference = 'Stop'
$RevisionRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RevisionOutput = Join-Path $RevisionRoot 'jt_oct\_native'
$RevisionVsWhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$RevisionInstall = & $RevisionVsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $RevisionInstall) { throw 'MSVC C++ toolchain is required' }
if (-not (Test-Path -LiteralPath (Join-Path $GurobiRoot 'include\gurobi_c++.h'))) { throw 'Gurobi C++ SDK is required' }
New-Item -ItemType Directory -Force -Path $RevisionOutput | Out-Null
$RevisionVcVars = Join-Path $RevisionInstall 'VC\Auxiliary\Build\vcvars64.bat'
$RevisionCommand = '"' + $RevisionVcVars + '" >nul && cl /nologo /O2 /MD /openmp /EHsc /std:c++17 /c /Fo"' + $RevisionOutput + '\revision_d3_optimized.obj" "' + $RevisionRoot + '\native\d3_optimized.cpp" && cl /nologo /O2 /MD /openmp /EHsc /std:c++17 /LD /I"' + $GurobiRoot + '\include" /Fo"' + $RevisionOutput + '\revision_experiments.obj" "' + $RevisionRoot + '\native\revision_experiments.cpp" "' + $RevisionOutput + '\revision_d3_optimized.obj" /link /LIBPATH:"' + $GurobiRoot + '\lib" gurobi_c++md2017.lib gurobi130.lib psapi.lib /OUT:"' + $RevisionOutput + '\revision_experiments.dll" /IMPLIB:"' + $RevisionOutput + '\revision_experiments.lib"'
cmd.exe /d /s /c $RevisionCommand
if ($LASTEXITCODE -ne 0) { throw 'Revision native build failed' }
Write-Output (Join-Path $RevisionOutput 'revision_experiments.dll')
