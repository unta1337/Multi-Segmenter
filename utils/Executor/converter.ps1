$folderPath = "assets/tests"

$serials = Get-ChildItem -Path $folderPath -Filter *serial*.txt
$parallels = Get-ChildItem -Path $folderPath -Filter *parallel*.txt

for ($i=0; $i -lt $serials.Count; $i++) {
    $serial = $serials[$i]
    node utils/LogTool/converter.js $serial report.csv
}

for ($i=0; $i -lt $parallels.Count; $i++) {
    $parallel = $parallels[$i]
    node utils/LogTool/converter.js $parallel report.csv
}