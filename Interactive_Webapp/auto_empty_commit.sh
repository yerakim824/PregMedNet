git commit --allow-empty -m "automate empty commit: $(date)"
git push origin main

# The job is automated with crontab to run every day at midnight. Check the status  by
# crontab -l
# OR
# crontab -e
