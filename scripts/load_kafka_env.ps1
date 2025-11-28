$envFile = "infra\kafka.env"

if (!(Test-Path $envFile)) {
    Write-Error "Kafka env file '$envFile' not found. 
Copy infra\kafka.env.example to infra\kafka.env and fill in your own values."
    exit 1
}

Write-Host "Loading Kafka env vars from $envFile`n"

Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -eq "" -or $line.StartsWith("#")) { return }

    $name, $value = $line -split '=', 2
    $name  = $name.Trim()
    $value = $value.Trim()

    [Environment]::SetEnvironmentVariable($name, $value, "Process")
    Write-Host "Set $name"
}

Write-Host "`nKafka environment variables are now set for this terminal session."
