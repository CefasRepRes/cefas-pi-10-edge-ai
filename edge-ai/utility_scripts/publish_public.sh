# Start fresh
rm -rf /tmp/pub_clean
mkdir -p /tmp/pub_clean

# Go to the private repo
cd /c/Users/JR13/Documents/LOCAL_NOT_ONEDRIVE/rapid-plankton

# Export tracked files (respects .gitignore)
git archive --format=tar HEAD | (cd /tmp/pub_clean && tar -xf -)

# Create a brand-new repository with no history
cd /tmp/pub_clean

git init
git add -A
git commit -m "Public release (commit history squashed for security purposes)"
git branch -M main

git remote add origin git@github.com:CefasRepRes/cefas-pi-10-edge-ai.git

git push --force origin main