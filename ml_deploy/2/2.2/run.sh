REPLICA_NAME_NEW=`python replica_name.py`
echo "result: ${REPLICA_NAME_NEW}"
export REPLICA_NAME=${REPLICA_NAME_NEW}
python main.py &
exec python gracefull_shutdown.py