[[ $# -eq 0 ]] && { echo "Usage: $0 tag"; exit 1; }

docker build  --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --tag $1 .
