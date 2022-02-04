
## Start contributing! (Pull Requests)

Before writing code, we strongly advise you to search through the existing PRs or
issues to make sure that nobody is already working on the same thing. If you are
unsure, it is always a good idea to open an issue to get some feedback.

You will need basic `git` proficiency to be able to contribute to
`bridge-inspection-toolkit`. `git` is not the easiest tool to use but it has the greatest
manual. Type `git --help` in a shell and enjoy. If you prefer books, [Pro
Git](https://git-scm.com/book/en/v2) is a very good reference.

Follow these steps to start contributing:

1. Fork the [repository](https://github.com/phiyodr/bridge-inspection-toolkit) by
   clicking on the 'Fork' button on the repository's page. This creates a copy of the code
   under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   $ git clone git@github.com:<YourGithubName>/bridge-inspection-toolkit.git
   $ cd bridge-inspection-toolkit
   $ git remote add upstream https://github.com/phiyodr/bridge-inspection-toolkit.git
   ```

3. Create a new branch to hold your development changes:

   ```bash
   $ git checkout -b a-descriptive-name-for-my-changes
   ```

   **Do not** work on the `master` branch.

4. Set up a development environment by running the following command in a virtual environment:

   ```bash
   $ pip install -e .
   ```

   (If bridge-inspection-toolkit was already installed in the virtual environment, remove
   it with `pip uninstall bikit` before reinstalling it in editable
   mode with the `-e` flag.)


5. Develop the features on your branch.

   As you work on the features, you should make sure that the test suite
   passes:

   ```bash
   # cd bridge-inspection-toolkit/
   pytest
   ```

   Once you're happy with your changes, add changed files using `git add` and
   make a commit with `git commit` to record your changes locally:

   ```bash
   $ git add modified_file.py
   $ git commit
   ```

   Please write [good commit
   messages](https://chris.beams.io/posts/git-commit/).

   It is a good idea to sync your copy of the code with the original
   repository regularly. This way you can quickly account for changes:

   ```bash
   $ git fetch upstream
   $ git rebase upstream/master
   ```

   Push the changes to your account using:

   ```bash
   $ git push -u origin a-descriptive-name-for-my-changes
   ```

6. Once you are satisfied (**and the checklist below is happy too**), go to the
   webpage of your fork on GitHub. Click on 'Pull request' to send your changes
   to the project maintainers for review.

7. So everyone can see the changes in the Pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.


### Checklist

1. The title of your pull request should be a summary of its contribution;
2. If your pull request addresses an issue, please mention the issue number in
   the pull request description to make sure they are linked (and people
   consulting the issue know you are working on it);
3. To indicate a work in progress please prefix the title with `[WIP]`. These
   are useful to avoid duplicated work, and to differentiate it from PRs ready
   to be merged;
4. Make sure existing tests pass;
6. All public methods must have informative docstrings.

### Tests

  Run tests with

   ```bash
   # cd bridge-inspection-toolkit/
   pytest
   ```


### Style guide

For documentation strings, `bikit` follows the [google style](https://google.github.io/styleguide/pyguide.html).


### Acknowledgement

#### This guide is heavily based on the awesome [transformers guide to contributing](https://github.com/huggingface/transformers/blob/master/CONTRIBUTING.md) which itself was heavily inspired by the [scikit-learn guide](https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md).

