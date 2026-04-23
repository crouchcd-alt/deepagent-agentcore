#!/usr/bin/env bash
raw_account_id="$1"

cat <<EOF
This script sets up Git filters to mask the AWS account ID in your repository.

When you run this script with your AWS account ID, it will configure Git to replace the actual account ID with a placeholder (AWS_ACCOUNT_ID_PLACEHOLDER) when you commit changes. When you check out files, it will replace the placeholder with the actual account ID from an environment variable.

You must:
- have .env.masked file with the line: AWS_ACCOUNT_ID=your_actual_account_id

EOF

# assert that AWS_ACCOUNT_ID exists in .env.masked within a subshell
if ! (source .env.masked 2>/dev/null && [[ -n "$AWS_ACCOUNT_ID" ]]); then
  echo "Error: AWS_ACCOUNT_ID not found in .env.masked. Please create the file with the line: AWS_ACCOUNT_ID=your_actual_account_id"
  exit 1
fi

git config filter.aws-account.clean  "bash -c 'source .env.masked 2>/dev/null; sed \"s/\$AWS_ACCOUNT_ID/AWS_ACCOUNT_ID_PLACEHOLDER/g\"'"
git config filter.aws-account.smudge "bash -c 'source .env.masked 2>/dev/null; sed \"s/AWS_ACCOUNT_ID_PLACEHOLDER/\$AWS_ACCOUNT_ID/g\"'"
