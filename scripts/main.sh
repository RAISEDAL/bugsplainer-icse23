if [ "$(basename "$PWD")" = 'bugsplainer-replication-package' ]; then
  printf "Replicating...\n";
else
  echo 'This script must be run from inside bugsplainer-replication-package directory';
  return 1;
fi

./_install.sh

printf '\n\nReplicating RQ1...\n\n'
./_rq1.1.sh
./_rq1.2.sh
./_rq1.3.sh
./_rq1.4.sh
