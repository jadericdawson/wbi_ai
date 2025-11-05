# Quick Start: Adding New Users to WBI AI

## 🚀 Fast Track (2 minutes)

Use the automated script to grant all necessary permissions:

```bash
./scripts/onboard_user.sh newuser@example.com
```

That's it! The script will:
- ✅ Grant Azure OpenAI access
- ✅ Grant Blob Storage access
- ✅ Provide email template to send to the new user

---

## 📖 Full Documentation

For detailed instructions, troubleshooting, and manual setup via Azure Portal:

**See:** [USER_ONBOARDING_GUIDE.md](./USER_ONBOARDING_GUIDE.md)

---

## ⚡ Quick Reference

### What the script does
Grants these RBAC roles to new users:

| Role | Resource | Purpose |
|------|----------|---------|
| Cognitive Services OpenAI User | jadericdawson-4245-resource | Access AI models |
| Storage Blob Data Contributor | wbiaistorage | Save chat history |

### After running the script
1. **Wait 5-10 minutes** for permissions to propagate
2. **Send user the app URL:** http://wbi-ai.azurewebsites.net
3. **User signs in** with their Azure AD credentials

### Troubleshooting
If user gets 401 errors:
- Wait 5-10 minutes for RBAC propagation
- Have user clear browser cache and sign out/in
- Verify roles were assigned: `az role assignment list --assignee <user_email> --output table`

---

## 🔐 Security Benefits

Using Azure AD authentication with RBAC:
- ✅ No shared API keys
- ✅ Centralized access control in Azure Portal
- ✅ Full audit trail
- ✅ Easy permission revocation
- ✅ Per-user authentication

---

## 📞 Support

- **Automation issues:** Check [scripts/onboard_user.sh](./scripts/onboard_user.sh)
- **Manual setup:** See [USER_ONBOARDING_GUIDE.md](./USER_ONBOARDING_GUIDE.md)
- **App issues:** Check application logs in Azure Portal

---

**Last Updated:** 2025-11-05
**Application:** http://wbi-ai.azurewebsites.net
