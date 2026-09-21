param([string]$GurobiRoot = $env:GUROBI_HOME)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$gurobi = if ($GurobiRoot) { $GurobiRoot } else { 'C:\gurobi1300\win64' }
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $install) { throw 'Visual Studio C++ toolchain not found' }
$dev = Join-Path $install 'Common7\Tools\VsDevCmd.bat'
$out = Join-Path $root 'jt_oct\_native'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$src = Join-Path $root 'native\deep_structural_core.cpp'
$oracle = Join-Path $root 'native\d3_optimized.cpp'
$dll = Join-Path $out 'deep_structural_core.dll'
$cmd = '"' + $dev + '" -arch=x64 && cl /nologo /O2 /MD /EHsc /openmp /std:c++17 /LD "' + $src + '" "' + $oracle + '" /I"' + $gurobi + '\include" /link /LIBPATH:"' + $gurobi + '\lib" gurobi_c++md2017.lib gurobi130.lib Psapi.lib /OUT:"' + $dll + '"'
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "Native deep structural build failed: $LASTEXITCODE" }
Write-Host $dll
