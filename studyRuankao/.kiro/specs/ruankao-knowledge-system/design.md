# Design Document

## Overview

The Ruankao Knowledge System is a comprehensive solution for software designer exam preparation, consisting of two integrated components:

1. **Knowledge Repository**: A structured collection of Markdown documents organized by exam modules
2. **Search Platform**: A web-based application providing search, filtering, and progress tracking capabilities

The system follows a content-first approach where the knowledge repository serves as the single source of truth, and the search platform provides an enhanced interface for content discovery and study management.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Search Platform (Web App)                │
├─────────────────────────────────────────────────────────────┤
│  Frontend (React/Vue)  │  Backend API (Node.js/Python)     │
│  - Search Interface    │  - Content Indexing               │
│  - Progress Tracking   │  - Search Engine                  │
│  - Responsive UI       │  - User Data Management           │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                   Knowledge Repository                      │
├─────────────────────────────────────────────────────────────┤
│  Structured Markdown Files                                 │
│  ├── modules/                                              │
│  │   ├── programming-languages/                           │
│  │   ├── data-structures-algorithms/                      │
│  │   ├── operating-systems/                               │
│  │   ├── database-systems/                                │
│  │   ├── software-engineering/                            │
│  │   ├── object-oriented-technology/                      │
│  │   ├── network-security/                                │
│  │   └── computer-organization/                           │
│  ├── metadata/                                             │
│  └── assets/                                               │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

- **Frontend**: React with TypeScript for type safety and better development experience
- **Backend**: Node.js with Express for API services
- **Search Engine**: Elasticsearch or Lunr.js for full-text search capabilities
- **Database**: SQLite for user data and progress tracking (lightweight and portable)
- **Content Processing**: Gray-matter for Markdown frontmatter parsing
- **Deployment**: Docker containers for easy deployment and scaling

## Components and Interfaces

### 1. Knowledge Repository Structure

#### Directory Organization
```
knowledge-repository/
├── modules/
│   ├── programming-languages/
│   │   ├── index.md
│   │   ├── language-processing.md
│   │   ├── program-calling.md
│   │   └── language-comparison.md
│   ├── data-structures-algorithms/
│   │   ├── index.md
│   │   ├── linear-structures.md
│   │   ├── trees-binary-trees.md
│   │   ├── graphs.md
│   │   └── algorithms.md
│   └── [other modules...]
├── metadata/
│   ├── module-config.json
│   └── link-registry.json
└── assets/
    ├── images/
    └── diagrams/
```

#### Markdown Document Template
```markdown
---
title: "Knowledge Point Title"
module: "module-name"
importance: "高|中|低"
frequency: "很高|高|中等"
tags: ["tag1", "tag2"]
related: ["related-topic-1", "related-topic-2"]
external_links:
  - url: "https://example.com"
    title: "External Resource Title"
    verified: true
    last_checked: "2024-12-15"
---

# Knowledge Point Title

## 核心概念 (Core Concepts)

## 重要细节 (Important Details)

## 考试要点 (Exam Focus)

## 相关链接 (Related Links)

## 练习题目 (Practice Questions)
```

### 2. Search Platform Components

#### Frontend Components
- **SearchInterface**: Main search input with autocomplete
- **FilterPanel**: Module, importance, and frequency filters
- **ResultsList**: Search results with highlighting
- **KnowledgeViewer**: Markdown content renderer
- **ProgressTracker**: Study progress visualization
- **NavigationMenu**: Module-based navigation

#### Backend API Endpoints
```
GET /api/search?q={query}&module={module}&importance={level}
GET /api/modules
GET /api/knowledge-points/{id}
POST /api/progress/mark-studied
GET /api/progress/stats
GET /api/content/validate-links
```

### 3. Content Processing Pipeline

#### Indexing Process
1. **Content Scanner**: Recursively scans repository for Markdown files
2. **Metadata Extractor**: Parses frontmatter and extracts structured data
3. **Content Parser**: Converts Markdown to searchable text
4. **Link Validator**: Verifies external links and updates status
5. **Search Indexer**: Updates search index with processed content

#### Real-time Updates
- File system watcher monitors repository changes
- Incremental indexing for modified files
- Cache invalidation for updated content
- WebSocket notifications for real-time UI updates

## Data Models

### Knowledge Point Model
```typescript
interface KnowledgePoint {
  id: string;
  title: string;
  module: string;
  importance: 'high' | 'medium' | 'low';
  frequency: 'very_high' | 'high' | 'medium';
  tags: string[];
  content: string;
  related: string[];
  external_links: ExternalLink[];
  file_path: string;
  last_updated: Date;
}

interface ExternalLink {
  url: string;
  title: string;
  verified: boolean;
  last_checked: Date;
}
```

### User Progress Model
```typescript
interface UserProgress {
  user_id: string;
  knowledge_point_id: string;
  status: 'not_started' | 'studying' | 'mastered';
  last_accessed: Date;
  study_time: number; // in minutes
}

interface StudyStats {
  total_points: number;
  studied_points: number;
  mastered_points: number;
  progress_by_module: Record<string, ModuleProgress>;
}
```

### Module Configuration Model
```typescript
interface ModuleConfig {
  id: string;
  name: string;
  display_name: string;
  description: string;
  importance_weight: number;
  total_knowledge_points: number;
  estimated_study_hours: number;
}
```

## Error Handling

### Content Validation
- **Missing Frontmatter**: Warn and use default values
- **Invalid Links**: Mark as broken and notify content managers
- **Duplicate IDs**: Generate unique identifiers automatically
- **Malformed Markdown**: Log errors and skip problematic sections

### Search Platform Errors
- **Search Service Down**: Fallback to basic text matching
- **Database Connection Issues**: Use local storage for progress tracking
- **Content Loading Failures**: Display cached content with update notifications
- **Network Timeouts**: Implement retry logic with exponential backoff

### User Experience
- Graceful degradation when services are unavailable
- Clear error messages with suggested actions
- Offline mode for previously accessed content
- Progress data synchronization when connection is restored

## Testing Strategy

### Content Testing
- **Link Validation**: Automated checks for external link availability
- **Content Structure**: Validate Markdown frontmatter and structure
- **Cross-Reference Integrity**: Ensure related links point to existing content
- **Performance Testing**: Measure indexing time for large content sets

### Platform Testing
- **Unit Tests**: Individual component functionality
- **Integration Tests**: API endpoints and database operations
- **End-to-End Tests**: Complete user workflows
- **Performance Tests**: Search response times and concurrent user handling
- **Accessibility Tests**: WCAG compliance for inclusive design

### Continuous Validation
- **Daily Link Checks**: Automated validation of external resources
- **Content Freshness**: Monitor for outdated information
- **Search Quality**: Track search result relevance and user satisfaction
- **Performance Monitoring**: Response times and system resource usage

## Security Considerations

### Data Protection
- No sensitive personal information stored
- Local storage for user preferences and progress
- HTTPS enforcement for all external communications
- Input sanitization for search queries

### Content Security
- Markdown sanitization to prevent XSS attacks
- Validation of external links before inclusion
- Rate limiting for search API endpoints
- Content versioning for audit trails

## Deployment and Scalability

### Development Environment
- Docker Compose for local development
- Hot reloading for content changes
- Integrated testing environment
- Documentation generation from code

### Production Deployment
- Containerized application deployment
- CDN for static assets and content
- Horizontal scaling for search services
- Automated backup for user progress data

### Performance Optimization
- Content caching strategies
- Search index optimization
- Lazy loading for large content sets
- Progressive web app capabilities for offline access