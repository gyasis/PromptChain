# Research Agent Version Tracking

## Current Version: 0.1.0.a

## Version Management System

This project uses semantic versioning with alpha character increments for progress tracking:

### Version Format: MAJOR.MINOR.PATCH.ALPHA

- **MAJOR**: Breaking changes requiring significant updates
- **MINOR**: New features that are backward compatible
- **PATCH**: Bug fixes and minor improvements  
- **ALPHA**: Work-in-progress increments (a, b, c, d, e, ...)

### Alpha Version Progression Example:
```
0.1.0     -> Initial stable release
0.1.0.a   -> First incremental change
0.1.0.b   -> Second incremental change  
0.1.0.c   -> Third incremental change
0.1.1     -> Next stable release (when feature is complete)
```

## Current Development Milestones

### Version 0.1.0.a - Project Initialization
- [x] Repository setup with security exclusions
- [x] Core project structure established
- [x] Dependencies configured in pyproject.toml
- [x] PRD documentation included
- [ ] Core modules implementation
- [ ] Testing framework setup
- [ ] CI/CD configuration

### Next Version: 0.1.0.b (Planned)
- [ ] Implement basic RAG functionality
- [ ] Literature search integration
- [ ] Configuration system activation

## Commit Message Conventions

Use conventional commits for clear history:

- `feat:` - New features
- `fix:` - Bug fixes  
- `docs:` - Documentation updates
- `style:` - Code formatting
- `refactor:` - Code restructuring
- `test:` - Testing additions
- `chore:` - Maintenance tasks

## Branch Strategy

- `main` - Production-ready code
- `feature/description` - New feature development
- `fix/description` - Bug fixes
- `hotfix/description` - Urgent fixes

## Version Update Process

1. Make incremental changes on feature branch
2. Update version in pyproject.toml (e.g., 0.1.0.a → 0.1.0.b)
3. Commit with descriptive message
4. When feature is complete, increment to next semantic version (0.1.1)
5. Merge to main with proper tagging

## Safety and Rollback

Each commit should be:
- **Atomic**: Single logical change
- **Reversible**: Safe to revert without breaking system
- **Tested**: Functionality verified before commit
- **Documented**: Clear commit message explaining changes

## Security Protocols

**NEVER commit these files:**
- `.env` files with API keys
- `.specstory/` folders  
- `claude.md` or `cursor.md` files
- Database files (*.db, *.sqlite)
- Cache directories
- IDE-specific configuration

These are automatically excluded via .gitignore.