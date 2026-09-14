param([string]$GurobiRoot = $env:GUROBI_HOME)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($GurobiRoot)) {
    $GurobiRoot = 'C:\gurobi1300\win64'
}

& (Join-Path $Root "build_native.ps1")
& (Join-Path $Root "build_d3_prepare.ps1")
& (Join-Path $Root "build_d3_optimized.ps1")
& (Join-Path $Root "build_d2_cg.ps1") -GurobiRoot $GurobiRoot
& (Join-Path $Root "build_d3_cg.ps1") -GurobiRoot $GurobiRoot
& (Join-Path $Root "build_contract_rmp.ps1") -GurobiRoot $GurobiRoot
Write-Output "Native JT-OCT backends built successfully."
