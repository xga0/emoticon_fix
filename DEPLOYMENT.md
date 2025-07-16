# PyPI Deployment Guide

This guide explains how to set up automatic deployment to PyPI using GitHub Actions.

## Prerequisites

1. A PyPI account at [https://pypi.org](https://pypi.org)
2. A Test PyPI account at [https://test.pypi.org](https://test.pypi.org) (optional but recommended)
3. Your package already registered on PyPI (which you already have: `emoticon-fix`)

## Setup Instructions

### 1. Generate PyPI API Tokens

#### For PyPI (Production)
1. Go to [https://pypi.org/manage/account/](https://pypi.org/manage/account/)
2. Scroll down to "API tokens" and click "Add API token"
3. Give it a name like "GitHub Actions - emoticon-fix"
4. Set scope to "Entire account" or limit to just your package
5. Copy the token (starts with `pypi-`)

#### For Test PyPI (Optional)
1. Go to [https://test.pypi.org/manage/account/](https://test.pypi.org/manage/account/)
2. Follow the same steps as above
3. Copy the token

### 2. Add Secrets to GitHub Repository

1. Go to your GitHub repository
2. Click "Settings" → "Secrets and variables" → "Actions"
3. Click "New repository secret"
4. Add these secrets:

   - **Name**: `PYPI_API_TOKEN`
   - **Value**: Your PyPI API token (including the `pypi-` prefix)

   - **Name**: `TEST_PYPI_API_TOKEN` (optional)
   - **Value**: Your Test PyPI API token

## How the Workflow Works

### Automatic Deployment (Recommended)
- **Trigger**: When you create a new GitHub release
- **Action**: Automatically builds and uploads to PyPI
- **Best for**: Production releases

### Manual Deployment
- **Trigger**: Manual workflow dispatch from GitHub Actions tab
- **Action**: Uploads to Test PyPI for testing
- **Best for**: Testing before official release

## Creating a Release

1. Update the version in `emoticon_fix/__init__.py`:
   ```python
   __version__ = "0.3.0"  # Your new version
   ```

2. Commit and push the changes:
   ```bash
   git add emoticon_fix/__init__.py
   git commit -m "Bump version to 0.3.0"
   git push
   ```

3. Create a new release on GitHub:
   - Go to your repository → "Releases" → "Create a new release"
   - Tag version: `v0.3.0` (matching your version)
   - Release title: `v0.3.0`
   - Add release notes describing changes
   - Click "Publish release"

4. The workflow will automatically trigger and deploy to PyPI!

## Testing the Workflow

### Test on Test PyPI First (Recommended)
1. Go to "Actions" tab in your repository
2. Click "Publish to PyPI" workflow
3. Click "Run workflow" → "Run workflow"
4. This will upload to Test PyPI where you can verify everything works

### Verify Deployment
- Check [https://pypi.org/project/emoticon-fix/](https://pypi.org/project/emoticon-fix/) for the new version
- Test installation: `pip install emoticon-fix==YOUR_NEW_VERSION`

## Troubleshooting

### Common Issues

1. **"Token is invalid"**
   - Verify your PyPI token is correct
   - Ensure it includes the `pypi-` prefix
   - Check that the token hasn't expired

2. **"Package already exists"**
   - You're trying to upload a version that already exists
   - Bump the version number in `__init__.py`

3. **"Insufficient permissions"**
   - Ensure your PyPI token has permissions for this package
   - If using a scoped token, verify it includes your package

4. **"Build failed"**
   - Check the Actions logs for detailed error messages
   - Common issues: missing dependencies, syntax errors

### Getting Help
- Check the Actions logs for detailed error messages
- PyPI documentation: [https://packaging.python.org/](https://packaging.python.org/)
- GitHub Actions documentation: [https://docs.github.com/en/actions](https://docs.github.com/en/actions)

## Workflow Features

- ✅ Builds using modern Python packaging (`python -m build`)
- ✅ Validates distribution before upload (`twine check`)
- ✅ Supports both PyPI and Test PyPI
- ✅ Manual and automatic triggers
- ✅ Uses secure API tokens (no passwords)
- ✅ Compatible with modern Python packaging standards

## Security Notes

- API tokens are stored as encrypted secrets in GitHub
- Tokens are only accessible during workflow execution
- Use scoped tokens when possible (limit to specific packages)
- Regularly rotate your API tokens 