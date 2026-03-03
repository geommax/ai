# 🔧 Popular MCP Servers & Tools (2026 ထိ)

> **Model Context Protocol (MCP)** သည် Anthropic မှ ဖန်တီးထားသော open protocol ဖြစ်ပြီး AI models များအား local နှင့် remote resources များနှင့် လုံခြုံစွာ ချိတ်ဆက်နိုင်စေပါသည်။

---

## 📑 မာတိကာ

- [🏛️ Official Reference Servers](#️-official-reference-servers)
- [🎖️ အသုံးအများဆုံး Official Integrations](#️-အသုံးအများဆုံး-official-integrations)
- [🌍 Community မှ Popular Servers](#-community-မှ-popular-servers)
- [🔗 Aggregators & Meta Servers](#-aggregators--meta-servers)
- [📂 Browser Automation](#-browser-automation)
- [🗄️ Databases](#️-databases)
- [💻 Developer Tools](#-developer-tools)
- [🔎 Search & Data Extraction](#-search--data-extraction)
- [🧠 Knowledge & Memory](#-knowledge--memory)
- [💬 Communication](#-communication)
- [💰 Finance & Fintech](#-finance--fintech)
- [📊 Monitoring & Observability](#-monitoring--observability)
- [🔒 Security](#-security)
- [☁️ Cloud Platforms](#️-cloud-platforms)
- [🔄 Version Control](#-version-control)
- [📂 File Systems](#-file-systems)
- [🛠️ MCP Ecosystem Tools & Registries](#️-mcp-ecosystem-tools--registries)
- [⚙️ Configuration Example](#️-configuration-example)

---

## 🏛️ Official Reference Servers

Anthropic / MCP Steering Group မှ တိုက်ရိုက် maintain လုပ်ထားသော reference servers များ —

| Server | Description | Language |
|--------|-------------|----------|
| **[Everything](https://github.com/modelcontextprotocol/servers/tree/main/src/everything)** | Reference / test server — prompts, resources, tools အားလုံးပါ | TypeScript |
| **[Fetch](https://github.com/modelcontextprotocol/servers/tree/main/src/fetch)** | Web content fetching & conversion — LLM-friendly output | Python |
| **[Filesystem](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem)** | File operations — configurable access controls | TypeScript |
| **[Git](https://github.com/modelcontextprotocol/servers/tree/main/src/git)** | Git repository read, search, analyze | Python |
| **[Memory](https://github.com/modelcontextprotocol/servers/tree/main/src/memory)** | Knowledge graph-based persistent memory | TypeScript |
| **[Sequential Thinking](https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking)** | Dynamic, reflective problem-solving | TypeScript |
| **[Time](https://github.com/modelcontextprotocol/servers/tree/main/src/time)** | Timezone conversion & time utilities | TypeScript |

---

## 🎖️ အသုံးအများဆုံး Official Integrations

ကုမ္ပဏီကြီးများမှ ၎င်းတို့၏ platforms အတွက် production-ready ဖြစ်အောင် maintain လုပ်ထားသော servers —

### 🔝 Top-Tier (အသုံးအများဆုံး)

| Server | Company | Description |
|--------|---------|-------------|
| **[GitHub](https://github.com/github/github-mcp-server)** | GitHub | Repo management, PRs, issues, code search — GitHub ၏ official MCP Server |
| **[Playwright](https://github.com/microsoft/playwright-mcp)** | Microsoft | Browser automation — accessibility snapshots, page navigation, screenshots |
| **[AWS](https://github.com/awslabs/mcp)** | Amazon | AWS best practices & development workflow servers |
| **[Cloudflare](https://github.com/cloudflare/mcp-server-cloudflare)** | Cloudflare | Workers, KV, R2, D1 — Cloudflare developer platform |
| **[Supabase](https://github.com/supabase-community/supabase-mcp)** | Supabase | Database, auth, storage management |
| **[Stripe](https://github.com/atharvagupta2003/mcp-stripe)** | Community | Payment, customer, refund management |
| **[Linear](https://linear.app/docs/mcp)** | Linear | Issue tracking, project management |
| **[Slack](https://github.com/korotovsky/slack-mcp-server)** | Community | Workspace messaging — Stdio & SSE transports |
| **[Notion](https://github.com/suekou/mcp-notion-server)** | Community | Notion API — pages, databases, search |
| **[Terraform](https://github.com/hashicorp/terraform-mcp-server)** | HashiCorp | IaC development — Terraform ecosystem integration |

### 🌟 Platform & Service Integrations

| Server | Description |
|--------|-------------|
| **[Tavily](https://github.com/tavily-ai/tavily-mcp)** | AI agent search engine — search + extract |
| **[Browserbase](https://github.com/browserbase/mcp-server-browserbase)** | Cloud browser automation — navigation, extraction, form filling |
| **[Apify](https://github.com/apify/apify-mcp-server)** | 6,000+ pre-built web scraping tools |
| **[BrightData](https://github.com/luminati-io/brightdata-mcp)** | Web access automation across the public internet |
| **[Elasticsearch](https://github.com/elastic/mcp-server-elasticsearch)** | Elasticsearch data query |
| **[Prisma](https://www.prisma.io/docs/postgres/integrations/mcp-server)** | Prisma Postgres database management |
| **[Perplexity](https://github.com/ppl-ai/modelcontextprotocol)** | Sonar API — real-time web-wide research |
| **[Composio](https://docs.composio.dev/docs/mcp-overview)** | 100+ tools — zero setup, auth built-in |
| **[Zapier](https://zapier.com/mcp)** | 8,000+ apps instant connection |
| **[Auth0](https://github.com/auth0/auth0-mcp-server)** | Identity & access management — actions, applications, forms |
| **[Langfuse](https://github.com/langfuse/mcp-server-langfuse)** | Prompt management — versioning, evaluating, releasing |
| **[Pinecone](https://github.com/pinecone-io/pinecone-mcp)** | Vector database — documentation search, index management |
| **[Qdrant](https://github.com/qdrant/mcp-server-qdrant/)** | Semantic memory layer — vector search engine |
| **[PayPal](https://mcp.paypal.com/)** | PayPal ၏ official MCP server |
| **[Razorpay](https://github.com/razorpay/razorpay-mcp-server)** | Razorpay ၏ official payment MCP server |
| **[Webflow](https://github.com/webflow/mcp-server)** | Sites, pages, collections management |
| **[WordPress](https://developer.wordpress.com/docs/mcp/)** | WordPress.com content, analytics, settings |

### 🗃️ Database Official Servers

| Server | Description |
|--------|-------------|
| **[MCP Toolbox for Databases](https://github.com/googleapis/genai-toolbox)** | Google — AlloyDB, BigQuery, Spanner, Postgres, MySQL, Neo4j + more |
| **[SQL Server (MSSQL)](https://github.com/Azure-Samples/SQL-AI-samples/tree/main/MssqlMcp)** | Microsoft ၏ official SQL Server MCP |
| **[MariaDB](https://github.com/mariadb/mcp)** | SQL + vector/embedding search |
| **[Redis Cloud](https://github.com/redis/mcp-redis-cloud/)** | Redis Cloud resource management |
| **[Milvus](https://github.com/zilliztech/mcp-server-milvus)** | Vector database search & interaction |
| **[Upstash](https://github.com/upstash/mcp-server)** | Redis management with natural language |
| **[MotherDuck](https://github.com/motherduckdb/mcp-server-motherduck)** | DuckDB query & analysis |
| **[Tinybird](https://github.com/tinybirdco/mcp-tinybird)** | Serverless ClickHouse platform |
| **[TiDB](https://github.com/pingcap/pytidb)** | TiDB database platform interaction |

---

## 🌍 Community မှ Popular Servers

### 📂 Browser Automation

| Server | ⭐ | Description |
|--------|----|-------------|
| **[Playwright (Community)](https://github.com/executeautomation/mcp-playwright)** | 🔥 | Browser automation & web scraping with Playwright |
| **[Puppeteer](https://github.com/modelcontextprotocol/servers-archived/tree/main/src/puppeteer)** | 🔥 | Browser automation — web scraping & interaction |
| **[BrowserMCP](https://github.com/browsermcp/mcp)** | 🔥 | Local Chrome browser automation |
| **[Skyvern](https://github.com/Skyvern-AI/skyvern/tree/main/integrations/mcp)** | ⭐ | LLM-controlled browser for Claude/Windsurf/Cursor |

### 🗄️ Databases

| Server | Description |
|--------|-------------|
| **[PostgreSQL (crystaldba)](https://github.com/crystaldba/postgres-mcp)** | All-in-one Postgres dev & operations — performance analysis, tuning |
| **[MySQL](https://github.com/designcomputer/mysql_mcp_server)** | MySQL with configurable access & schema inspection |
| **[SQLite (MCP Reference)](https://github.com/modelcontextprotocol/servers-archived)** | SQLite — archived reference |
| **[MongoDB](https://github.com/kiliczsh/mcp-mongo-server)** | MongoDB integration |
| **[Snowflake](https://github.com/Snowflake-Labs/mcp)** | Cortex Agents, structured/unstructured data, SQL execution |
| **[Chroma](https://github.com/chroma-core/chroma-mcp)** | Vector database — retrieval capabilities |
| **[ClickHouse](https://github.com/ClickHouse/mcp-clickhouse)** | Schema inspection & query |
| **[DuckDB](https://github.com/ktanaka101/mcp-server-duckdb)** | DuckDB integration |
| **[Weaviate](https://github.com/weaviate/mcp-server-weaviate)** | Knowledge base & chat memory store |
| **[Neo4j (via cognee)](https://github.com/topoteretes/cognee/tree/main/cognee-mcp)** | GraphRAG memory — knowledge graph |

### 💻 Developer Tools

| Server | Description |
|--------|-------------|
| **[DesktopCommander](https://github.com/wonderwhy-er/DesktopCommanderMCP)** | File management, terminal commands, SSH — most popular local MCP |
| **[Docker](https://github.com/ckreiling/mcp-server-docker)** | Container, image, volume, network management |
| **[Docker Hub (Official)](https://github.com/docker/hub-mcp)** | Docker Hub search & repository management |
| **[JetBrains](https://github.com/JetBrains/mcpProxy)** | JetBrains IDE connection |
| **[Postman](https://github.com/delano/postman-mcp-server)** | Postman API interaction |
| **[MCP Installer](https://github.com/anaisbetts/mcp-installer)** | Installs other MCP servers for you |
| **[Language Server](https://github.com/isaacphi/mcp-language-server)** | Semantic tools — get definition, references, rename, diagnostics |
| **[Shrimp Task Manager](https://github.com/cjo4m06/mcp-shrimp-task-manager)** | Programming-focused task management for AI coding agents |
| **[Context7](https://github.com/augmnt/augments-mcp-server)** | 90+ framework docs — React, Next.js, Laravel, FastAPI etc. |
| **[Sentry](https://github.com/getsentry/sentry-mcp)** | Error tracking & performance monitoring |
| **[Next.js DevTools](https://github.com/vercel/next-devtools-mcp)** | Official Next.js MCP — runtime diagnostics, route inspection |

### 🔎 Search & Data Extraction

| Server | Description |
|--------|-------------|
| **[Brave Search](https://github.com/brave/brave-search-mcp-server)** | Brave Search API — web search |
| **[Exa](https://github.com/exa-labs/exa-mcp-server)** | AI Search API — real-time web information |
| **[DuckDuckGo](https://github.com/nickclyde/duckduckgo-mcp-server)** | Free web search — no API key |
| **[Kagi](https://github.com/kagisearch/kagimcp)** | Official Kagi Search MCP |
| **[SerpApi](https://github.com/serpapi/serpapi-mcp)** | Multi-engine search — Google, Bing, Yahoo, YouTube, eBay + more |
| **[SearXNG](https://github.com/ihor-sokoliuk/mcp-searxng)** | Self-hosted meta search engine |
| **[GitMCP](https://github.com/idosal/git-mcp)** | Connect to ANY GitHub repo documentation |
| **[RAG Web Browser (Apify)](https://github.com/apify/mcp-server-rag-web-browser)** | Web search + scrape + Markdown output |
| **[Olostep](https://github.com/olostep/olostep-mcp-server)** | Web scraping, crawling, search — AI answers with citations |
| **[ArXiv](https://github.com/blazickjp/arxiv-mcp-server)** | ArXiv research paper search |

### 🧠 Knowledge & Memory

| Server | Description |
|--------|-------------|
| **[Memory (Official)](https://github.com/modelcontextprotocol/servers-archived/tree/main/src/memory)** | Knowledge graph-based persistent memory |
| **[Obsidian (calclavia)](https://github.com/calclavia/mcp-obsidian)** | Markdown notes / Obsidian vault read & search |
| **[Mem0](https://github.com/mem0ai/mem0-mcp)** | Coding preferences & patterns management |
| **[mcp-memory-service](https://github.com/doobidoo/mcp-memory-service)** | Semantic memory — persistent storage across 13+ AI apps |
| **[Cognee](https://github.com/topoteretes/cognee/tree/dev/cognee-mcp)** | Memory manager — graph & vector stores, 30+ data sources |
| **[Zotero](https://github.com/kaliaboi/mcp-zotero)** | Research library management |
| **[RAG (Minima)](https://github.com/dmayboroda/minima)** | Local file RAG |

### 💬 Communication & Productivity

| Server | Description |
|--------|-------------|
| **[Slack](https://github.com/korotovsky/slack-mcp-server)** | Full Slack Workspace integration — no admin bot required |
| **[Discord](https://github.com/SaseQ/discord-mcp)** | Comprehensive Discord integration |
| **[WhatsApp](https://github.com/lharries/whatsapp-mcp)** | Personal WhatsApp — search, send messages |
| **[Telegram](https://github.com/chaindead/telegram-mcp)** | Telegram API bridge — data, dialogs, messages |
| **[Gmail / Google Calendar](https://github.com/MarkusPfundstein/mcp-gsuite)** | Google Workspace integration |
| **[Microsoft Teams](https://github.com/InditexTech/mcp-teams-server)** | Teams messaging — read, post, mention |
| **[Atlassian (Jira + Confluence)](https://github.com/sooperset/mcp-atlassian)** | Confluence & Jira — search, read, create, manage |
| **[Todoist](https://github.com/abhiz123/todoist-mcp-server)** | Task management |

### 💰 Finance & Fintech

| Server | Description |
|--------|-------------|
| **[Financial Datasets](https://github.com/financial-datasets/mcp-server)** | Stock market API for AI agents |
| **[Twelve Data](https://github.com/twelvedata/mcp)** | Real-time & historical financial market data |
| **[CoinGecko](https://github.com/coingecko/coingecko-typescript/tree/main/packages/mcp-server)** | Crypto price & market data — 200+ blockchains, 8M+ tokens |
| **[Alpaca](https://github.com/alpacahq/alpaca-mcp-server)** | Stock & options trading via Alpaca API |
| **[Polygon.io](https://github.com/polygon-io/mcp_polygon)** | Stocks, indices, forex, options data |

### 📊 Monitoring & Observability

| Server | Description |
|--------|-------------|
| **[Grafana](https://github.com/grafana/mcp-grafana)** | Dashboards, incidents, datasources |
| **[Sentry](https://github.com/getsentry/sentry-mcp)** | Error tracking & performance monitoring |
| **[Datadog](https://github.com/TANTIOPE/datadog-mcp-server)** | Logs, APM traces, metrics |
| **[Logfire (Pydantic)](https://github.com/pydantic/logfire-mcp)** | OpenTelemetry traces & metrics |
| **[PostHog](https://github.com/posthog/mcp)** | Analytics, feature flags, error tracking |

### 🔒 Security

| Server | Description |
|--------|-------------|
| **[Semgrep](https://github.com/semgrep/mcp)** | Code vulnerability scanning |
| **[GitGuardian](https://github.com/GitGuardian/gg-mcp)** | 500+ secret detectors — credential leak prevention |
| **[GhidraMCP](https://github.com/LaurieWired/GhidraMCP)** | Reverse engineering — LLM-autonomous binary analysis |
| **[Burp Suite](https://github.com/PortSwigger/mcp-server)** | Web application security testing |
| **[StackHawk](https://github.com/stackhawk/stackhawk-mcp)** | Security testing for vibe-coded apps |
| **[BoostSecurity](https://github.com/boost-community/boost-mcp)** | Dependency vulnerability, malware, typosquatting detection |
| **[VirusTotal](https://github.com/BurtTheCoder/mcp-virustotal)** | URL scanning, file hash analysis, IP reports |
| **[Shodan](https://github.com/BurtTheCoder/mcp-shodan)** | IP lookups, device searches, vulnerability queries |

### ☁️ Cloud Platforms

| Server | Description |
|--------|-------------|
| **[AWS (Official)](https://github.com/awslabs/mcp)** | AWS best practices for development workflow |
| **[Azure (Official)](https://github.com/Azure/azure-mcp)** | Storage, Cosmos DB, Azure Monitor |
| **[Cloudflare](https://github.com/cloudflare/mcp-server-cloudflare)** | Workers, KV, R2, D1 |
| **[Pulumi](https://github.com/pulumi/mcp-server)** | Cloud infrastructure deployment & management |
| **[Render](https://render.com/docs/mcp-server)** | Service spin-up, database queries, metrics |
| **[Kubernetes (mcp-k8s-go)](https://github.com/strowk/mcp-k8s-go)** | Pod browsing, logs, events, namespaces |
| **[LocalStack](https://github.com/localstack/localstack-mcp-server)** | Local AWS environments management |

### 🔄 Version Control

| Server | Description |
|--------|-------------|
| **[GitHub (Official)](https://github.com/github/github-mcp-server)** | Full GitHub API — repos, PRs, issues, search |
| **[GitLab](https://github.com/modelcontextprotocol/servers-archived/tree/main/src/gitlab)** | GitLab platform — project management, CI/CD |
| **[Bitbucket](https://github.com/JaviMaligno/mcp-server-bitbucket)** | 58 tools — repos, PRs, pipelines, branches |
| **[Azure DevOps](https://github.com/Tiberriver256/mcp-server-azure-devops)** | Repos, work items, pipelines |

### 📂 File Systems

| Server | Description |
|--------|-------------|
| **[Filesystem (Official)](https://github.com/modelcontextprotocol/servers-archived/tree/main/src/filesystem)** | Local file system with secure access |
| **[Google Drive](https://github.com/isaacphi/mcp-gdrive)** | Google Drive read & Sheets editing |
| **[Box](https://github.com/box/mcp-server-box-remote/)** | Secure Box content access — search, Q&A, extraction |
| **[MarkItDown](https://github.com/microsoft/markitdown/tree/main/packages/markitdown-mcp)** | Microsoft — file format to Markdown conversion |
| **[Pandoc](https://github.com/vivekVells/mcp-pandoc)** | Document format conversion — MD, HTML, PDF, DOCX + more |

---

## 🔗 Aggregators & Meta Servers

တစ်နေရာတည်းမှ servers များစွာကို ချိတ်ဆက်/ရှာဖွေ/စီမံနိုင်သော aggregator servers —

| Server | Description |
|--------|-------------|
| **[Composio](https://docs.composio.dev/docs/mcp-overview)** | 100+ tools — zero setup |
| **[Zapier](https://zapier.com/mcp)** | 8,000+ apps connection |
| **[Pipedream](https://github.com/PipedreamHQ/pipedream/tree/master/modelcontextprotocol)** | 2,500 APIs, 8,000+ prebuilt tools |
| **[MetaMCP](https://github.com/metatool-ai/metatool-app)** | Unified middleware — GUI-based MCP management |
| **[Plugged.in](https://github.com/VeriTeknik/pluggedin-mcp)** | Multiple MCP servers → single MCP proxy |
| **[WayStation](https://github.com/waystation-ai/mcp)** | Notion, Slack, Monday, AirTable — 90 sec setup |
| **[MCP Installer](https://github.com/anaisbetts/mcp-installer)** | Auto-install other MCP servers |

---

## 🛠️ MCP Ecosystem Tools & Registries

MCP servers များကို ရှာဖွေ/install/manage လုပ်ရန် tools များ —

| Tool | Description |
|------|-------------|
| **[MCP Registry](https://registry.modelcontextprotocol.io/)** | Official MCP Server Registry |
| **[Smithery](https://smithery.ai/)** | MCP server registry — find tools for LLM agents |
| **[Glama.ai](https://glama.ai/mcp/servers)** | Awesome MCP Servers web directory |
| **[mcp-get](https://mcp-get.com/)** | CLI tool — install & manage MCP servers |
| **[mcpm.sh](https://mcpm.sh/)** | Homebrew-like MCP Manager |
| **[Toolbase](https://gettoolbase.ai/)** | Desktop app — no-code MCP management |
| **[ToolHive](https://github.com/StacklokLabs/toolhive)** | Containerized MCP server deployment |
| **[MCP Linker](https://github.com/milisp/mcp-linker)** | Cross-platform GUI — Claude, Cursor, VS Code, Neovim |
| **[MCP Inspector](https://glama.ai/mcp/inspector)** | Test & inspect MCP servers |
| **[MCPHub](https://www.mcphub.com/)** | High quality MCP servers listing with real user reviews |
| **[PulseMCP](https://www.pulsemcp.com/)** | Weekly newsletter + community hub |
| **[mcp.run](https://mcp.run/)** | Hosted registry & control plane |

---

## ⚙️ Configuration Example

Claude Desktop (သို့) MCP Client တွင် servers configure လုပ်ရန် —

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "<YOUR_TOKEN>"
      }
    },
    "playwright": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-playwright"]
    }
  }
}
```

### Quick Install Commands

```bash
# TypeScript-based servers
npx -y @modelcontextprotocol/server-memory
npx -y @modelcontextprotocol/server-filesystem /path/to/allowed/files

# Python-based servers
uvx mcp-server-git
uvx mcp-server-fetch
```

---

## 📊 Key Stats (2026 ခုနှစ်အထိ)

| Metric | Value |
|--------|-------|
| **Official Reference Servers** | 7 |
| **Official Integrations** | 300+ |
| **Community Servers** | 2,000+ |
| **Awesome MCP Servers ⭐** | 82,100+ |
| **Contributors** | 1,058+ |
| **Categories** | 25+ |
| **Supported Languages** | TypeScript, Python, Go, Rust, C#, Java, Ruby, C/C++ |

---

## 📚 Resources & Learning

| Resource | Link |
|----------|------|
| MCP Official Docs | https://modelcontextprotocol.io/ |
| MCP GitHub Org | https://github.com/modelcontextprotocol |
| Awesome MCP Servers | https://github.com/punkpeye/awesome-mcp-servers |
| MCP Discord | https://glama.ai/mcp/discord |
| Reddit r/mcp | https://www.reddit.com/r/mcp |
| MCP Quickstart Guide | https://glama.ai/blog/2024-11-25-model-context-protocol-quickstart |

---

> **Last Updated:** March 2026  
> **Sources:** [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers), [punkpeye/awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers), [modelcontextprotocol.io](https://modelcontextprotocol.io/)
