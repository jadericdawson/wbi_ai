#!/bin/bash

# ========================================
# WBI AI User Onboarding Script
# ========================================
# This script grants the necessary Azure RBAC permissions for new users
# to access the WBI AI application.
#
# Usage:
#   ./onboard_user.sh user@example.com
#
# Or run interactively:
#   ./onboard_user.sh
# ========================================

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SUBSCRIPTION_ID="4151f5f9-5d87-40e7-8329-4921512a08ee"
OPENAI_RESOURCE_GROUP="rg-jadericdawson-4245"
OPENAI_RESOURCE_NAME="jadericdawson-4245-resource"
STORAGE_RESOURCE_GROUP="rg-wbi-ai-eastus2"
STORAGE_ACCOUNT_NAME="wbiaistorage"
APP_URL="http://wbi-ai.azurewebsites.net"

# Functions
print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  WBI AI User Onboarding${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    print_error "Azure CLI is not installed. Please install it first:"
    echo "  https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Check if logged in to Azure
if ! az account show &> /dev/null; then
    print_error "Not logged in to Azure CLI"
    echo ""
    echo "Please log in first:"
    echo "  az login"
    exit 1
fi

# Print header
print_header

# Get user email
if [ -z "$1" ]; then
    echo "Enter the new user's email address (Azure AD account):"
    read -p "Email: " NEW_USER_EMAIL
else
    NEW_USER_EMAIL="$1"
fi

# Validate email format
if [[ ! "$NEW_USER_EMAIL" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
    print_error "Invalid email format: $NEW_USER_EMAIL"
    exit 1
fi

echo ""
print_info "Setting up access for: $NEW_USER_EMAIL"
echo ""

# Set subscription context
print_info "Setting Azure subscription context..."
az account set --subscription $SUBSCRIPTION_ID
print_success "Subscription set"

# Get user's object ID
print_info "Looking up user in Azure AD..."
USER_OBJECT_ID=$(az ad user show --id "$NEW_USER_EMAIL" --query id -o tsv 2>/dev/null)

if [ -z "$USER_OBJECT_ID" ]; then
    print_error "User not found in Azure AD: $NEW_USER_EMAIL"
    echo ""
    echo "Possible reasons:"
    echo "  - User doesn't exist in your Azure AD tenant"
    echo "  - Email address is incorrect"
    echo "  - User needs to be invited as a guest user first"
    echo ""
    echo "To invite a guest user:"
    echo "  az ad user create --user-principal-name $NEW_USER_EMAIL --display-name \"Guest User\" --force-change-password-next-login"
    exit 1
fi

print_success "Found user (Object ID: $USER_OBJECT_ID)"
echo ""

# Grant Azure OpenAI Access
print_info "Granting Azure OpenAI access..."
OPENAI_SCOPE="/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$OPENAI_RESOURCE_GROUP/providers/Microsoft.CognitiveServices/accounts/$OPENAI_RESOURCE_NAME"

if az role assignment create \
    --role "Cognitive Services OpenAI User" \
    --assignee "$USER_OBJECT_ID" \
    --scope "$OPENAI_SCOPE" \
    &> /dev/null; then
    print_success "Azure OpenAI access granted"
else
    # Check if already exists
    EXISTING=$(az role assignment list \
        --assignee "$USER_OBJECT_ID" \
        --scope "$OPENAI_SCOPE" \
        --role "Cognitive Services OpenAI User" \
        --query "[0].id" -o tsv 2>/dev/null)

    if [ -n "$EXISTING" ]; then
        print_warning "Azure OpenAI access already exists (skipping)"
    else
        print_error "Failed to grant Azure OpenAI access"
        echo "Please check Azure permissions and try again"
        exit 1
    fi
fi

# Grant Blob Storage Access
print_info "Granting Blob Storage access..."
STORAGE_SCOPE="/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$STORAGE_RESOURCE_GROUP/providers/Microsoft.Storage/storageAccounts/$STORAGE_ACCOUNT_NAME"

if az role assignment create \
    --role "Storage Blob Data Contributor" \
    --assignee "$USER_OBJECT_ID" \
    --scope "$STORAGE_SCOPE" \
    &> /dev/null; then
    print_success "Blob Storage access granted"
else
    # Check if already exists
    EXISTING=$(az role assignment list \
        --assignee "$USER_OBJECT_ID" \
        --scope "$STORAGE_SCOPE" \
        --role "Storage Blob Data Contributor" \
        --query "[0].id" -o tsv 2>/dev/null)

    if [ -n "$EXISTING" ]; then
        print_warning "Blob Storage access already exists (skipping)"
    else
        print_error "Failed to grant Blob Storage access"
        echo "Please check Azure permissions and try again"
        exit 1
    fi
fi

# Success message
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ✅ User Onboarding Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "📧 Send the following information to $NEW_USER_EMAIL:"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Subject: Access Granted - WBI AI Application"
echo ""
echo "Hi,"
echo ""
echo "You now have access to the WBI AI application!"
echo ""
echo "Application URL: $APP_URL"
echo "Sign in with your Azure AD credentials: $NEW_USER_EMAIL"
echo ""
echo "Features available:"
echo "  • Multi-Agent Team RAG queries"
echo "  • Chat with AI persona"
echo "  • Document upload & knowledge base search"
echo "  • Chat history saved automatically"
echo ""
echo "⚠️  Important: RBAC permissions may take 5-10 minutes to propagate."
echo "If you get authentication errors, please wait a few minutes and try again."
echo ""
echo "Questions? Contact your administrator."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Show verification commands
echo -e "${BLUE}ℹ To verify permissions were granted:${NC}"
echo ""
echo "  az role assignment list --assignee $USER_OBJECT_ID --output table"
echo ""

print_warning "Note: Permissions may take 5-10 minutes to propagate through Azure"
echo ""
