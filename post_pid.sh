SERIES_JSON=$(curl -s -X GET "http://13.66.246.174/series/get?patient_id=${1}" -H "accept: application/json" -u test:test)
RESULT=$(curl -s -X POST "http://52.250.112.18:8888/inference" -H "Content-Type: application/json" -d "${SERIES_JSON}")
echo $RESULT
