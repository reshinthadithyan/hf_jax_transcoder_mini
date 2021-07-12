### Usage

1. Go to ```download_repo_info.py``` and set USER={your-github-user-name} and TOKEN={your-github-access-token}

2. Now run the following command to download the required repos, prepare and process the data:

```
git clone https://github.com/tree-sitter/tree-sitter-cpp

git clone https://github.com/tree-sitter/tree-sitter-java.git

make prepare-data
```