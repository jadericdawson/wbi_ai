# WBI AI Multi-User Onboarding Guide

## Overview
This guide provides step-by-step instructions for adding new users to the WBI AI application with proper Azure RBAC permissions.

## Prerequisites
- Azure subscription: `4151f5f9-5d87-40e7-8329-4921512a08ee`
- Azure CLI installed and authenticated
- User must have an Azure AD account

## Required Resources

The application uses the following Azure resources:

| Resource | Type | Resource Group | Purpose |
|----------|------|----------------|---------|
| jadericdawson-4245-resource | Azure OpenAI | rg-jadericdawson-4245 | LLM models (GPT-4.1, DeepSeek-R1) |
| wbiaistorage | Storage Account | rg-wbi-ai-eastus2 | User data and chat history |
| wbi-ai-cosmos | Cosmos DB | rg-wbi-ai-eastus2 | Knowledge base (uses key auth) |
| wbi-ai | Web App | rg-knowledge2ai-eastus | The Streamlit application |

## Step-by-Step User Onboarding

### Option 1: Using Azure Portal (Recommended for Non-Technical Users)

#### 1. Grant Azure OpenAI Access

1. Navigate to the Azure Portal: https://portal.azure.com
2. Go to **Resource Groups** → **rg-jadericdawson-4245**
3. Click on **jadericdawson-4245-resource** (Cognitive Services account)
4. In the left menu, click **Access control (IAM)**
5. Click **+ Add** → **Add role assignment**
6. Select role: **Cognitive Services OpenAI User**
7. Click **Next**
8. Click **+ Select members**
9. Search for the user's email address
10. Click **Select**
11. Click **Review + assign** twice

#### 2. Grant Blob Storage Access

1. Go to **Resource Groups** → **rg-wbi-ai-eastus2**
2. Click on **wbiaistorage** (Storage account)
3. In the left menu, click **Access control (IAM)**
4. Click **+ Add** → **Add role assignment**
5. Select role: **Storage Blob Data Contributor**
6. Click **Next**
7. Click **+ Select members**
8. Search for the user's email address
9. Click **Select**
10. Click **Review + assign** twice

#### 3. Test Access

Have the user:
1. Navigate to: http://wbi-ai.azurewebsites.net
2. Sign in with their Azure AD credentials (if prompted)
3. Try using the Multi-Agent Team feature
4. If successful, they should see agent responses without 401 errors

---

### Option 2: Using Azure CLI (Recommended for Automation)

#### Quick Start Script

Replace `user@example.com` with the new user's email address:

```bash
#!/bin/bash

# Configuration
NEW_USER_EMAIL="user@example.com"
SUBSCRIPTION_ID="4151f5f9-5d87-40e7-8329-4921512a08ee"

# Set subscription context
az account set --subscription $SUBSCRIPTION_ID

# Get user's object ID
USER_OBJECT_ID=$(az ad user show --id $NEW_USER_EMAIL --query id -o tsv)

if [ -z "$USER_OBJECT_ID" ]; then
    echo "❌ Error: User $NEW_USER_EMAIL not found in Azure AD"
    exit 1
fi

echo "✓ Found user: $NEW_USER_EMAIL (Object ID: $USER_OBJECT_ID)"

# 1. Grant Azure OpenAI Access
echo "⏳ Granting Azure OpenAI access..."
az role assignment create \
    --role "Cognitive Services OpenAI User" \
    --assignee $USER_OBJECT_ID \
    --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/rg-jadericdawson-4245/providers/Microsoft.CognitiveServices/accounts/jadericdawson-4245-resource"

if [ $? -eq 0 ]; then
    echo "✓ Azure OpenAI access granted"
else
    echo "❌ Failed to grant Azure OpenAI access"
fi

# 2. Grant Blob Storage Access
echo "⏳ Granting Blob Storage access..."
az role assignment create \
    --role "Storage Blob Data Contributor" \
    --assignee $USER_OBJECT_ID \
    --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/rg-wbi-ai-eastus2/providers/Microsoft.Storage/storageAccounts/wbiaistorage"

if [ $? -eq 0 ]; then
    echo "✓ Blob Storage access granted"
else
    echo "❌ Failed to grant Blob Storage access"
fi

echo ""
echo "✅ User onboarding complete!"
echo "📧 Send the following info to $NEW_USER_EMAIL:"
echo ""
echo "   Application URL: http://wbi-ai.azurewebsites.net"
echo "   Sign in with your Azure AD credentials"
echo ""
echo "⚠️  Note: It may take 5-10 minutes for permissions to propagate."
```

#### Manual Commands

If you prefer to run commands individually:

```bash
# Set variables
NEW_USER_EMAIL="user@example.com"
SUBSCRIPTION_ID="4151f5f9-5d87-40e7-8329-4921512a08ee"

# Get user object ID
USER_OBJECT_ID=$(az ad user show --id $NEW_USER_EMAIL --query id -o tsv)
echo "User Object ID: $USER_OBJECT_ID"

# Grant Azure OpenAI access
az role assignment create \
    --role "Cognitive Services OpenAI User" \
    --assignee $USER_OBJECT_ID \
    --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/rg-jadericdawson-4245/providers/Microsoft.CognitiveServices/accounts/jadericdawson-4245-resource"

# Grant Blob Storage access
az role assignment create \
    --role "Storage Blob Data Contributor" \
    --assignee $USER_OBJECT_ID \
    --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/rg-wbi-ai-eastus2/providers/Microsoft.Storage/storageAccounts/wbiaistorage"
```

---

## Troubleshooting

### Issue: User gets 401 Unauthorized error

**Symptoms:**
```
Error code: 401 - {'error': {'code': 'Unauthorized', 'message': 'Access denied due to invalid subscription key or wrong API endpoint...'}}
```

**Solution:**
1. Verify user has **Cognitive Services OpenAI User** role on `jadericdawson-4245-resource`
2. Wait 5-10 minutes for RBAC propagation
3. Have user clear browser cache and sign out/in again
4. Check role assignments:
   ```bash
   az role assignment list --assignee $USER_OBJECT_ID --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/rg-jadericdawson-4245/providers/Microsoft.CognitiveServices/accounts/jadericdawson-4245-resource"
   ```

### Issue: User cannot access chat history

**Symptoms:**
- Error accessing blob storage
- Chat history not loading

**Solution:**
1. Verify user has **Storage Blob Data Contributor** role on `wbiaistorage`
2. Check role assignments:
   ```bash
   az role assignment list --assignee $USER_OBJECT_ID --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/rg-wbi-ai-eastus2/providers/Microsoft.Storage/storageAccounts/wbiaistorage"
   ```

### Issue: Permissions taking too long to propagate

**Solution:**
- RBAC propagation can take 5-10 minutes
- Have user sign out of Azure AD and sign back in
- Clear browser cache
- Wait up to 15 minutes in rare cases

---

## RBAC Roles Explained

### Cognitive Services OpenAI User
- **Purpose:** Allows the user to make API calls to Azure OpenAI models
- **Permissions:**
  - Read model deployments
  - Create chat completions
  - Use embeddings
- **Scope:** `jadericdawson-4245-resource` (Azure OpenAI account)

### Storage Blob Data Contributor
- **Purpose:** Allows read/write access to blob storage for user data
- **Permissions:**
  - Read blobs
  - Write blobs
  - List containers
- **Scope:** `wbiaistorage` (Storage account)

### Note on Cosmos DB
The application uses **key-based authentication** for Cosmos DB (via `COSMOS_KEY` environment variable), so no per-user RBAC is needed.

---

## Verifying User Access

To verify a user has been properly configured:

```bash
# Check all role assignments for a user
az role assignment list --assignee $USER_OBJECT_ID --output table

# Should show:
# - "Cognitive Services OpenAI User" on jadericdawson-4245-resource
# - "Storage Blob Data Contributor" on wbiaistorage
```

---

## Removing User Access

To revoke access for a user:

```bash
# Remove Azure OpenAI access
az role assignment delete \
    --role "Cognitive Services OpenAI User" \
    --assignee $USER_OBJECT_ID \
    --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/rg-jadericdawson-4245/providers/Microsoft.CognitiveServices/accounts/jadericdawson-4245-resource"

# Remove Blob Storage access
az role assignment delete \
    --role "Storage Blob Data Contributor" \
    --assignee $USER_OBJECT_ID \
    --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/rg-wbi-ai-eastus2/providers/Microsoft.Storage/storageAccounts/wbiaistorage"
```

---

## Security Notes

1. **Least Privilege:** Users only get access to resources they need
2. **No Keys Shared:** Users authenticate with their own Azure AD credentials
3. **Audit Trail:** All actions are logged in Azure Activity Log
4. **Centralized Control:** All access managed through Azure RBAC
5. **Easy Revocation:** Remove access instantly by revoking role assignments

---

## Questions?

If you encounter issues not covered in this guide:

1. Check Azure Activity Log for RBAC errors
2. Verify user is in the correct Azure AD tenant
3. Ensure subscription has sufficient quota for Azure OpenAI
4. Contact your Azure administrator

---

**Last Updated:** 2025-11-05
**Application:** WBI AI Multi-Agent RAG System
**URL:** http://wbi-ai.azurewebsites.net
