#wget "https://files.pushshift.io/reddit/comments/RC_"$1".bz2"
curl "https://files.pushshift.io/reddit/comments/RC_"$1".bz2" --output "RC_"$1".bz2"
bzip2 -d "RC_"$1".bz2"
