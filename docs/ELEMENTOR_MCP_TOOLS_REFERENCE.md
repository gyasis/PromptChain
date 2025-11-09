# Elementor MCP Server - Complete Tools Reference

## Overview

The **Elementor MCP Server** provides 7 powerful tools for managing WordPress pages with Elementor page builder integration. This server enables AI assistants to perform full CRUD (Create, Read, Update, Delete) operations on WordPress pages with Elementor data.

**Server Name:** `WordPressElementorMCP`  
**Version:** 1.0.0  
**Repository:** https://github.com/aguaitech/Elementor-MCP

---

## Prerequisites

Before using these tools, ensure:

1. **WordPress Site**: You have a WordPress site with REST API enabled
2. **Elementor Installed**: The Elementor plugin is installed and active
3. **Application Password**: Created in WordPress dashboard (Users → Profile → Application Passwords)
4. **MCP Configuration**: Proper environment variables set (`WP_URL`, `WP_APP_USER`, `WP_APP_PASSWORD`)

---

## All Available Tools

### 1. 🆕 `create_page`

**Description:** Creates a new WordPress page with Elementor data.

**Purpose:** Build new pages programmatically with custom Elementor layouts, sections, columns, and widgets.

**Parameters:**
- `title` (string, **required**): The title for the new page
- `elementor_data` (string, **required**): Elementor page data as a JSON string
- `status` (enum, optional): Page status - one of:
  - `"publish"` - Make page live immediately
  - `"draft"` - Save as draft (default)
  - `"pending"` - Pending review
  - `"private"` - Private page
  - `"future"` - Schedule for future publication
- `content` (string, optional): Standard WordPress content (HTML/text)

**Returns:**
- The newly created page ID (as string)

**Example Use Cases:**
- Creating landing pages from templates
- Building product pages programmatically
- Generating campaign-specific pages
- Bulk page creation from data

**Example:**
```json
{
  "title": "New Product Launch Page",
  "status": "draft",
  "content": "<p>Welcome to our new product!</p>",
  "elementor_data": "[{\"id\":\"abc123\",\"elType\":\"section\",...}]"
}
```

**Output:**
```
"42"  // New page ID
```

---

### 2. 📖 `get_page`

**Description:** Retrieves complete page data from WordPress by ID, including all Elementor metadata.

**Purpose:** Inspect existing pages, extract Elementor layouts, analyze page structure.

**Parameters:**
- `pageId` (number, **required**): The ID of the page to retrieve (must be positive integer)

**Returns:**
- Complete page object including:
  - Basic page data (title, content, status, slug, etc.)
  - Meta fields (including `_elementor_data`)
  - Page settings and configurations

**Example Use Cases:**
- Analyzing existing page structure
- Extracting Elementor layouts for reuse
- Auditing page configurations
- Debugging page issues

**Example:**
```json
{
  "pageId": 42
}
```

**Output:**
```json
{
  "id": 42,
  "title": { "rendered": "Product Launch Page" },
  "content": { "rendered": "<p>Welcome...</p>" },
  "status": "publish",
  "meta": {
    "_elementor_data": "[{\"id\":\"abc123\",...}]",
    "_elementor_edit_mode": "builder",
    "_elementor_template_type": "wp-page"
  }
}
```

---

### 3. 💾 `download_page_to_file`

**Description:** Downloads page data from WordPress and saves it to a local file.

**Purpose:** Backup pages, version control Elementor layouts, export for migration.

**Parameters:**
- `pageId` (number, **required**): The ID of the page to download
- `filePath` (string, **required**): Absolute path where to save the file (e.g., `/home/user/pages/backup.json`)
- `onlyElementorData` (boolean, optional, default: `false`): 
  - `true` - Save only the `_elementor_data` field
  - `false` - Save complete page object

**Returns:**
- `"true"` on success

**Example Use Cases:**
- Backing up page designs before major changes
- Version controlling Elementor layouts in Git
- Exporting pages for migration to another site
- Creating page templates library

**Example:**
```json
{
  "pageId": 42,
  "filePath": "/home/gyasis/Documents/elementor-backups/page-42.json",
  "onlyElementorData": true
}
```

**Output:**
```
"true"
```

**File Created:**
```json
[
  {
    "id": "abc123",
    "elType": "section",
    "settings": {...},
    "elements": [...]
  }
]
```

---

### 4. ✏️ `update_page`

**Description:** Updates an existing WordPress page with new data (title, content, status, or Elementor data).

**Purpose:** Modify existing pages programmatically, change layouts, update content.

**Parameters:**
- `pageId` (number, **required**): The ID of the page to update
- `title` (string, optional): New title for the page
- `status` (enum, optional): New status (`"publish"`, `"draft"`, `"pending"`, `"private"`, `"future"`)
- `content` (string, optional): New WordPress content (HTML/text)
- `elementor_data` (string, optional): New Elementor page data as JSON string

**Returns:**
- `"true"` on success

**Notes:**
- At least one update field must be provided
- Only provided fields will be updated (others remain unchanged)
- Elementor data must be valid JSON string

**Example Use Cases:**
- A/B testing different layouts
- Batch updating page statuses
- Programmatic content updates
- Applying design changes across multiple pages

**Example:**
```json
{
  "pageId": 42,
  "title": "Updated Product Launch Page",
  "status": "publish",
  "elementor_data": "[{\"id\":\"xyz789\",\"elType\":\"section\",...}]"
}
```

**Output:**
```
"true"
```

---

### 5. 📂 `update_page_from_file`

**Description:** Updates a WordPress page using Elementor data loaded from a local file.

**Purpose:** Restore backups, apply saved templates, batch page updates from file-based workflows.

**Parameters:**
- `pageId` (number, **required**): The ID of the page to update
- `elementorFilePath` (string, **required**): Absolute path to file containing Elementor data JSON
- `title` (string, optional): New title for the page
- `status` (enum, optional): New status (`"publish"`, `"draft"`, etc.)
- `contentFilePath` (string, optional): Absolute path to file containing WordPress content

**Returns:**
- `"true"` on success

**Example Use Cases:**
- Restoring pages from backups
- Deploying version-controlled page designs
- Applying template changes across multiple pages
- Migration workflows from development to production

**Example:**
```json
{
  "pageId": 42,
  "title": "Restored Product Page",
  "status": "publish",
  "elementorFilePath": "/home/gyasis/Documents/elementor-backups/page-42.json",
  "contentFilePath": "/home/gyasis/Documents/content/page-42.html"
}
```

**Output:**
```
"true"
```

---

### 6. 🗑️ `delete_page`

**Description:** Deletes a WordPress page, with option to bypass trash.

**Purpose:** Remove unwanted pages, clean up test content, bulk deletion operations.

**Parameters:**
- `pageId` (number, **required**): The ID of the page to delete
- `force` (boolean, optional, default: `false`):
  - `false` - Move to trash (can be restored)
  - `true` - Permanently delete (cannot be undone)

**Returns:**
- `"true"` on success

**⚠️ Warning:**
- Using `force: true` permanently deletes the page with no way to recover
- Always backup important pages before deletion

**Example Use Cases:**
- Removing outdated campaign pages
- Cleaning up test/development pages
- Bulk deletion of unused content
- Automated site maintenance

**Example (move to trash):**
```json
{
  "pageId": 42,
  "force": false
}
```

**Example (permanent deletion):**
```json
{
  "pageId": 42,
  "force": true
}
```

**Output:**
```
"true"
```

---

### 7. 🔍 `get_page_id_by_slug`

**Description:** Finds a WordPress page ID using its slug (URL-friendly name).

**Purpose:** Locate pages by their URL slug instead of numeric ID.

**Parameters:**
- `slug` (string, **required**): The slug (URL-friendly name) of the page
  - Example: `"about-us"` from URL `https://example.com/about-us`

**Returns:**
- The page ID (as string)

**Example Use Cases:**
- Finding pages by their URL
- Locating pages when ID is unknown
- URL-based page operations
- Content management workflows using slugs

**Example:**
```json
{
  "slug": "product-launch-2025"
}
```

**Output:**
```
"42"
```

---

## Common Workflows

### Workflow 1: Backup and Restore a Page

**Step 1: Backup (Download)**
```json
// Download page to file
{
  "tool": "download_page_to_file",
  "pageId": 42,
  "filePath": "/backups/page-42-backup.json",
  "onlyElementorData": false
}
```

**Step 2: Restore (Upload)**
```json
// Restore from backup
{
  "tool": "update_page_from_file",
  "pageId": 42,
  "elementorFilePath": "/backups/page-42-backup.json"
}
```

---

### Workflow 2: Clone a Page Design

**Step 1: Get original page**
```json
{
  "tool": "get_page",
  "pageId": 42
}
// Extract elementor_data from response
```

**Step 2: Create new page with same design**
```json
{
  "tool": "create_page",
  "title": "New Page with Same Design",
  "status": "draft",
  "elementor_data": "<extracted_from_step_1>"
}
```

---

### Workflow 3: Find and Update Page by URL

**Step 1: Get page ID from slug**
```json
{
  "tool": "get_page_id_by_slug",
  "slug": "about-us"
}
// Returns: "42"
```

**Step 2: Update the page**
```json
{
  "tool": "update_page",
  "pageId": 42,
  "title": "About Us - Updated",
  "status": "publish"
}
```

---

### Workflow 4: Version Control Page Designs

**Development Phase:**
```json
// Download page design
{
  "tool": "download_page_to_file",
  "pageId": 42,
  "filePath": "/git-repo/pages/homepage-v1.2.json",
  "onlyElementorData": true
}
// Commit to Git
```

**Deployment Phase:**
```json
// Deploy to production
{
  "tool": "update_page_from_file",
  "pageId": 123,  // Production page ID
  "elementorFilePath": "/git-repo/pages/homepage-v1.2.json",
  "status": "publish"
}
```

---

## Understanding Elementor Data Structure

Elementor pages are stored as JSON arrays containing nested elements:

```json
[
  {
    "id": "unique-id-1",
    "elType": "section",      // Element type: section, column, widget
    "settings": {             // Element settings
      "layout": "full_width",
      "background_color": "#ffffff"
    },
    "elements": [             // Child elements
      {
        "id": "unique-id-2",
        "elType": "column",
        "settings": {},
        "elements": [
          {
            "id": "unique-id-3",
            "elType": "widget",
            "widgetType": "heading",
            "settings": {
              "title": "Welcome!"
            }
          }
        ]
      }
    ]
  }
]
```

**Key Concepts:**
- **Sections**: Top-level containers (full-width, boxed, etc.)
- **Columns**: Divide sections into layout columns
- **Widgets**: Actual content elements (text, images, buttons, etc.)
- **Settings**: Configuration for each element (colors, spacing, animations, etc.)

---

## Error Handling

All tools throw errors that will be caught by the MCP server. Common errors:

**Authentication Errors:**
```
Error: Authentication failed. Check WP_APP_USER and WP_APP_PASSWORD.
```
**Solution:** Verify credentials in environment variables

**Page Not Found:**
```
Error: Page with ID 42 not found.
```
**Solution:** Verify page ID exists in WordPress

**Invalid JSON:**
```
Error: Invalid JSON in elementor_data parameter.
```
**Solution:** Ensure elementor_data is properly formatted JSON string

**File Not Found:**
```
Error: ENOENT: no such file or directory
```
**Solution:** Use absolute paths and ensure files exist

**Permission Errors:**
```
Error: User does not have permission to edit this page.
```
**Solution:** Ensure WordPress user has proper permissions (Editor or Administrator role)

---

## Best Practices

### 1. Always Backup Before Major Changes
```json
// Backup before updating
download_page_to_file(pageId, backup_path)
// Then update
update_page(pageId, new_data)
```

### 2. Use Draft Status for Testing
```json
{
  "status": "draft"  // Test changes before publishing
}
```

### 3. Validate JSON Before Sending
```javascript
// Ensure valid JSON
const data = JSON.stringify(elementorDataObject);
```

### 4. Use Absolute Paths for Files
```json
{
  "filePath": "/home/gyasis/backups/page.json"  // ✅ Good
  // NOT: "~/backups/page.json"  // ❌ Bad
}
```

### 5. Store Elementor Data Separately
```json
{
  "onlyElementorData": true  // Smaller, cleaner files for version control
}
```

---

## Security Considerations

### Application Password Best Practices

1. **Create Specific Passwords**: Create separate application passwords for different uses
2. **Limit Permissions**: Use WordPress user with minimum required permissions
3. **Rotate Regularly**: Change application passwords periodically
4. **Revoke Unused**: Delete application passwords that are no longer needed

### Environment Variable Security

```json
// Store in mcp.json (protected file)
"env": {
  "WP_URL": "https://www.twicedata.com",
  "WP_APP_USER": "api_user",
  "WP_APP_PASSWORD": "xxxx xxxx xxxx xxxx"  // Keep the spaces!
}
```

**Never:**
- Commit credentials to Git
- Share application passwords in plain text
- Use admin accounts for API access
- Store passwords in code

---

## Performance Tips

### 1. Batch Operations
Group multiple operations together rather than individual calls.

### 2. Use File-Based Updates for Large Changes
```json
// More efficient for large Elementor data
update_page_from_file(pageId, filePath)
```

### 3. Download Only What You Need
```json
{
  "onlyElementorData": true  // Smaller payload
}
```

### 4. Cache Page IDs from Slugs
If repeatedly accessing same pages, cache the ID lookup results.

---

## Troubleshooting Guide

### Problem: "No server info found" error

**Solution:**
1. Check MCP configuration in `~/.cursor/mcp.json`
2. Verify all environment variables are set
3. Ensure Node.js path is correct
4. Restart Cursor/Claude

### Problem: Authentication fails

**Solution:**
1. Verify WordPress username (not display name)
2. Check application password (keep spaces!)
3. Ensure WordPress REST API is enabled
4. Test credentials with curl:
```bash
curl -u "username:xxxx xxxx xxxx" https://yoursite.com/wp-json/wp/v2/pages
```

### Problem: Page not updating

**Solution:**
1. Check page ID is correct
2. Verify user has edit permissions
3. Ensure Elementor data is valid JSON
4. Check WordPress error logs

### Problem: File operations fail

**Solution:**
1. Use absolute paths (not `~` or relative)
2. Verify file/directory exists
3. Check file permissions
4. Ensure sufficient disk space

---

## Integration with AI Workflows

### Example: AI-Powered Page Generator

```
User: "Create a product landing page for our new smartwatch"

AI uses:
1. create_page() - Creates new draft page
2. get_page() - Retrieves page for confirmation
3. update_page() - Updates with AI-generated Elementor layout
4. download_page_to_file() - Backups the design
5. Final: update_page(status="publish") - Makes it live
```

### Example: Design System Propagation

```
User: "Update all product pages with new brand colors"

AI uses:
1. get_page_id_by_slug() - Finds each product page
2. get_page() - Retrieves current design
3. AI modifies colors in elementor_data
4. update_page() - Applies changes to each page
```

---

## Additional Resources

- **WordPress REST API Docs**: https://developer.wordpress.org/rest-api/
- **Elementor Developer Docs**: https://developers.elementor.com/
- **MCP Protocol Spec**: https://modelcontextprotocol.io/
- **Application Passwords Guide**: https://make.wordpress.org/core/2020/11/05/application-passwords-integration-guide/
- **Elementor Template Project**: https://github.com/aguaitech/Elementor_Project_Workflow

---

## Quick Reference Table

| Tool | Action | Input | Output | Use When |
|------|--------|-------|--------|----------|
| `create_page` | Create | title, elementor_data, status | Page ID | Building new pages |
| `get_page` | Read | pageId | Full page data | Inspecting pages |
| `download_page_to_file` | Read + Save | pageId, filePath | "true" | Backing up |
| `update_page` | Update | pageId, fields | "true" | Modifying pages |
| `update_page_from_file` | Update + Load | pageId, filePath | "true" | Restoring backups |
| `delete_page` | Delete | pageId, force | "true" | Removing pages |
| `get_page_id_by_slug` | Lookup | slug | Page ID | Finding by URL |

---

**Created:** October 11, 2025  
**Last Updated:** October 11, 2025  
**MCP Server Version:** 1.0.0  
**Documentation Version:** 1.0

