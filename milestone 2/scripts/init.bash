# - Load env variables into current environment

Get-Content .\infra\kafka.env | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
  $name,$value = $_ -split '=',2
  [Environment]::SetEnvironmentVariable($name.Trim(),$value.Trim(),'Process')
}

# Install dependencies from requirements.txt

pip install -r requirements.txt

# - Download kcat image

docker run --rm -it edenhill/kcat:1.7.1 -h

# - Test connection to kafka cluster/Test Metadata

docker run --rm -it edenhill/kcat:1.7.1 kcat -L -b "$($env:KAFKA_BOOTSTRAP)" -X security.protocol=SASL_SSL -X sasl.mechanisms=PLAIN -X sasl.username="$($env:KAFKA_API_KEY)" -X sasl.password="$($env:KAFKA_API_SECRET)"

# - Produce a test message to aerosparks.watch

'{"event":"ping","ts":"' + (Get-Date).ToUniversalTime().ToString("s") + 'Z"}' | docker run --rm -i edenhill/kcat:1.7.1 -b "$env:KAFKA_BOOTSTRAP" -X security.protocol=SASL_SSL -X sasl.mechanisms=PLAIN -X sasl.username="$env:KAFKA_API_KEY" -X sasl.password="$env:KAFKA_API_SECRET" -t aerosparks.watch -P

# - Consume last 10

docker run --rm -i edenhill/kcat:1.7.1 -b "$env:KAFKA_BOOTSTRAP" -X security.protocol=SASL_SSL -X sasl.mechanisms=PLAIN -X sasl.username="$env:KAFKA_API_KEY" -X sasl.password="$env:KAFKA_API_SECRET" -t aerosparks.watch -C -o -10 -e -q