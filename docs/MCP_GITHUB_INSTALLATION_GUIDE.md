# Complete Guide: Installing MCP Servers from GitHub (Not Published to npm)

## The Problem

Many MCP (Model Context Protocol) servers are hosted on GitHub but **NOT published to npm**. When you try to use them with `npx`, you get errors like:

```
npm error code ENOVERSIONS
npm error No versions available for <package-name>
```

OR

```
Error: Cannot find module '@modelcontextprotocol/sdk/server/mcp.js'
```

This guide shows you how to properly install and configure GitHub-only MCP servers.

---

## Solution Overview

Instead of using `npx` (which only works for npm-published packages), you need to:

1. **Clone the GitHub repository**
2. **Install dependencies locally**
3. **Configure MCP to run the local copy**

---

## Step-by-Step Installation Process

### Step 1: Identify the GitHub Repository

When you find an MCP server you want to use, note:
- The GitHub URL (e.g., `https://github.com/aguaitech/Elementor-MCP`)
- The entry point file (usually `src/index.js` or `dist/index.js`)

### Step 2: Clone the Repository

```bash
cd ~/Documents/code
git clone https://github.com/USERNAME/PROJECT-NAME.git
```

**Example:**
```bash
cd ~/Documents/code
git clone https://github.com/aguaitech/Elementor-MCP.git
```

### Step 3: Install Dependencies

```bash
cd PROJECT-NAME
npm install
# OR if using your specific NVM node version:
/home/gyasis/.nvm/versions/node/v22.9.0/bin/npm install
```

**Example:**
```bash
cd Elementor-MCP
/home/gyasis/.nvm/versions/node/v22.9.0/bin/npm install
```

**Important:** You should see output like:
```
added 99 packages, and audited 100 packages in 4s
```

### Step 4: Verify the Project Structure

Check where the main entry point is located:

```bash
ls -la src/
# OR
ls -la dist/
```

Common entry points:
- `src/index.js`
- `dist/index.js`
- `build/index.js`
- `index.js` (root level)

---

## Configuring MCP in Cursor/Claude

### Location of Configuration File

Your MCP configuration is in:
```
~/.cursor/mcp.json
```

### Configuration Pattern for Local GitHub MCP Servers

**Template:**
```json
{
  "mcpServers": {
    "YOUR-SERVER-NAME": {
      "command": "/home/gyasis/.nvm/versions/node/v22.9.0/bin/node",
      "args": ["/home/gyasis/Documents/code/PROJECT-NAME/PATH-TO-ENTRY-FILE"],
      "env": {
        "ENV_VAR_1": "value1",
        "ENV_VAR_2": "value2"
      }
    }
  }
}
```

### Real Example: Elementor MCP Server

**WRONG Configuration (using npx - doesn't work!):**
```json
{
  "mcpServers": {
    "elementor-wordpress": {
      "command": "/home/gyasis/.nvm/versions/node/v22.9.0/bin/npx",
      "args": ["-y", "elementor-mcp"],  ❌ Package not in npm!
      "env": {
        "WP_URL": "https://www.example.com",
        "WP_APP_USER": "username",
        "WP_APP_PASSWORD": "password"
      }
    }
  }
}
```

**CORRECT Configuration (using local clone):**
```json
{
  "mcpServers": {
    "elementor-wordpress": {
      "command": "/home/gyasis/.nvm/versions/node/v22.9.0/bin/node",
      "args": ["/home/gyasis/Documents/code/Elementor-MCP/src/index.js"],
      "env": {
        "WP_URL": "https://www.twicedata.com",
        "WP_APP_USER": "user",
        "WP_APP_PASSWORD": "Lvyo aW0t 3Wtt HeHx BJKk mhCW"
      }
    }
  }
}
```

---

## Key Configuration Rules

### 1. Use Full Paths (Always!)

✅ **CORRECT:**
```json
"command": "/home/gyasis/.nvm/versions/node/v22.9.0/bin/node"
"args": ["/home/gyasis/Documents/code/PROJECT-NAME/src/index.js"]
```

❌ **WRONG:**
```json
"command": "node"  // Might use wrong Node version
"args": ["~/Documents/code/PROJECT-NAME/src/index.js"]  // ~ doesn't expand
```

### 2. Match the README Environment Variables Exactly

Check the project's README.md for required environment variables and **use the exact names**:

**Example from Elementor-MCP README:**
```json
"env": {
  "WP_URL": "https://url.of.target.website",
  "WP_APP_USER": "wordpress_username",
  "WP_APP_PASSWORD": "Appl icat ion_ Pass word"
}
```

**Common mistakes:**
- Using `WORDPRESS_BASE_URL` instead of `WP_URL`
- Using `WORDPRESS_USERNAME` instead of `WP_APP_USER`
- Adding extra variables not in the documentation

### 3. Verify Your Node Version Path

Find your active Node version:
```bash
which node
# Output: /home/gyasis/.nvm/versions/node/v22.9.0/bin/node
```

Use this **exact path** in your `command` field.

---

## Testing Your Configuration

### Step 1: Restart Cursor/Claude

After editing `~/.cursor/mcp.json`, you **must restart** for changes to take effect:

1. Close Cursor completely
2. Reopen Cursor
3. The MCP server should auto-start

### Step 2: Check MCP Server Logs

MCP server logs are usually in:
```
~/.cursor/logs/
```

Look for your server name in the logs to see connection status and errors.

### Step 3: Test Basic Functionality

Try using an MCP tool from your server to verify it's working.

---

## Common Issues and Solutions

### Issue 1: "Cannot find module"

**Error:**
```
Error: Cannot find module '@modelcontextprotocol/sdk/server/mcp.js'
```

**Solution:**
You forgot to run `npm install` in the project directory:
```bash
cd /home/gyasis/Documents/code/PROJECT-NAME
npm install
```

### Issue 2: "Command not found"

**Error:**
```
Error: Command '/path/to/node' not found
```

**Solution:**
Your Node path is wrong. Verify it:
```bash
which node
# Use the output in your config
```

### Issue 3: "No server info found"

**Error in logs:**
```
[error] No server info found
```

**Solution:**
- Check that the entry point path is correct
- Verify environment variables are set correctly
- Check the project's README for required setup steps

### Issue 4: npm Vulnerabilities Warning

**Warning:**
```
2 vulnerabilities (1 high, 1 critical)
To address all issues, run: npm audit fix
```

**Solution (optional):**
```bash
cd /home/gyasis/Documents/code/PROJECT-NAME
npm audit fix
```

---

## Complete Working Examples

### Example 1: Elementor MCP Server

**Installation:**
```bash
cd ~/Documents/code
git clone https://github.com/aguaitech/Elementor-MCP.git
cd Elementor-MCP
npm install
```

**Configuration (~/.cursor/mcp.json):**
```json
{
  "mcpServers": {
    "elementor-wordpress": {
      "command": "/home/gyasis/.nvm/versions/node/v22.9.0/bin/node",
      "args": ["/home/gyasis/Documents/code/Elementor-MCP/src/index.js"],
      "env": {
        "WP_URL": "https://www.twicedata.com",
        "WP_APP_USER": "user",
        "WP_APP_PASSWORD": "Lvyo aW0t 3Wtt HeHx BJKk mhCW"
      }
    }
  }
}
```

### Example 2: n8n MCP Server (Already working in your setup)

**Installation:**
```bash
cd ~/Documents/code/n8n/mcp
git clone https://github.com/USERNAME/n8n-mcp-original.git
cd n8n-mcp-original
npm install
npm run build  # If TypeScript project
```

**Configuration:**
```json
{
  "mcpServers": {
    "n8n-mcp-working": {
      "command": "/home/gyasis/.nvm/versions/node/v22.16.0/bin/node",
      "args": ["/home/gyasis/Documents/code/n8n/mcp/n8n-mcp-original/dist/mcp/index.js"],
      "env": {
        "MCP_MODE": "stdio",
        "N8N_API_URL": "http://localhost:5678",
        "N8N_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

---

## Quick Reference Checklist

When installing any GitHub MCP server:

- [ ] Clone the repository to `~/Documents/code/`
- [ ] Run `npm install` in the project directory
- [ ] Find the entry point file (usually `src/index.js` or `dist/index.js`)
- [ ] Check README for required environment variables
- [ ] Get your full Node.js path: `which node`
- [ ] Edit `~/.cursor/mcp.json` with correct paths
- [ ] Use **full absolute paths** (not `~` or relative paths)
- [ ] Match environment variable names **exactly** from README
- [ ] Restart Cursor/Claude completely
- [ ] Check logs for connection status
- [ ] Test basic functionality

---

## Why This Happens

**npm vs GitHub:**
- `npx` only works with packages published to npm registry
- Many MCP servers are:
  - Open source projects on GitHub
  - Not published to npm (requires maintenance)
  - Available only as source code

**The Solution:**
- Clone and install locally
- Run directly with Node.js
- Configure with absolute paths

---

## Advanced: Keeping GitHub MCP Servers Updated

To update a GitHub MCP server:

```bash
cd ~/Documents/code/PROJECT-NAME
git pull origin main  # or master
npm install  # Install any new dependencies
```

Then restart Cursor/Claude to use the updated version.

---

## Summary

**For npm-published packages (use npx):**
```json
"command": "npx",
"args": ["-y", "package-name"]
```

**For GitHub-only packages (use local clone):**
```json
"command": "/full/path/to/node",
"args": ["/full/path/to/project/src/index.js"]
```

**Always:**
- Use full absolute paths
- Match environment variables from README exactly
- Run `npm install` after cloning
- Restart Cursor after config changes

---

## Need Help?

If you encounter issues:

1. Check the project's GitHub Issues page
2. Verify all paths are absolute and correct
3. Ensure `npm install` completed successfully
4. Check Cursor MCP logs for specific errors
5. Verify your Node version is compatible (usually v18+)

---

**Created:** October 11, 2025  
**Last Updated:** October 11, 2025  
**Tested With:** Node.js v22.9.0, Cursor IDE, Elementor-MCP

