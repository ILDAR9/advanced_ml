CONTAINER_ID=$(docker ps --filter "name=uwsgi_app" -q)
echo CONTAINER_ID: $CONTAINER_ID
docker exec  $CONTAINER_ID sh -c "touch /var/reloadFile"