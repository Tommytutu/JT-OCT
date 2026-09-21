param([string]$GurobiRoot = 'C:\gurobi1300\win64')
$ErrorActionPreference='Stop'
$DsRoot=Split-Path -Parent $MyInvocation.MyCommand.Path
$DsOutput=Join-Path $DsRoot 'jt_oct\_native'
$DsVsWhere='C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$DsInstall=& $DsVsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if(-not $DsInstall){throw 'MSVC C++ tools were not found'}
if(-not(Test-Path -LiteralPath(Join-Path $GurobiRoot 'include\gurobi_c++.h'))){throw 'Gurobi C++ SDK was not found'}
New-Item -ItemType Directory -Path $DsOutput -Force|Out-Null
$DsVcVars=Join-Path $DsInstall 'VC\Auxiliary\Build\vcvars64.bat'
$DsCommand='"'+$DsVcVars+'" >nul && cl /nologo /O2 /MD /openmp /EHsc /std:c++17 /LD /I"'+$GurobiRoot+'\include" /Fo"'+$DsOutput+'\d3_structural.obj" "'+$DsRoot+'\native\d3_structural.cpp" /link /LIBPATH:"'+$GurobiRoot+'\lib" gurobi_c++md2017.lib gurobi130.lib /OUT:"'+$DsOutput+'\d3_structural.dll" /IMPLIB:"'+$DsOutput+'\d3_structural.lib"'
cmd.exe /d /s /c $DsCommand
if($LASTEXITCODE -ne 0){throw 'D3 structural native build failed'}
Write-Output(Join-Path $DsOutput 'd3_structural.dll')
