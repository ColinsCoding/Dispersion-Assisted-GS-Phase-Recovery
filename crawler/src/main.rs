//! jalali-crawler
//!
//! Pulls papers from:
//!   1. Semantic Scholar API  (free, no key needed up to 100 req/s)
//!   2. arXiv API             (free, XML feed)
//!   3. Optional: direct URL list from a .txt file
//!
//! Stores results in SQLite (FTS5) at ../references/refs.db
//!
//! Binary tree (BST) on DOI for deduplication — O(log n) insert/lookup.
//!
//! Usage:
//!   cargo run --release -- --query "time-stretch dispersive Fourier" --limit 40
//!   cargo run --release -- --arxiv "jalali photonics" --limit 20
//!   cargo run --release -- --urls urls.txt

use anyhow::{Context, Result};
use clap::Parser;
use colored::*;
use regex::Regex;
use rusqlite::{params, Connection};
use scraper::{Html, Selector};
use serde::Deserialize;
use std::collections::BTreeMap;

// ── CLI ───────────────────────────────────────────────────────────────────────
#[derive(Parser, Debug)]
#[command(name = "jalali-crawler", about = "Photonics paper crawler → refs.db")]
struct Args {
    /// Semantic Scholar keyword query
    #[arg(long)]
    query: Option<String>,

    /// arXiv keyword query
    #[arg(long)]
    arxiv: Option<String>,

    /// Plain-text file of URLs to scrape (one per line)
    #[arg(long)]
    urls: Option<String>,

    /// Max papers per source
    #[arg(long, default_value_t = 40)]
    limit: usize,

    /// Path to SQLite database
    #[arg(long, default_value = "../references/refs.db")]
    db: String,

    /// Print full abstract for each result
    #[arg(long)]
    verbose: bool,
}

// ── Data model ────────────────────────────────────────────────────────────────
#[derive(Debug, Clone)]
struct Paper {
    doi:      String,
    title:    String,
    authors:  String,
    year:     i32,
    abstract_: String,
    source:   String,
    url:      String,
}

// ── Binary-tree deduplicator (BST on doi) ─────────────────────────────────────
/// BTreeMap<doi, Paper> gives O(log n) insert/lookup — the "binary tree for
/// photonics dataset" the user asked for.  Keys are canonicalised DOIs.
struct PaperTree {
    inner: BTreeMap<String, Paper>,
}

impl PaperTree {
    fn new() -> Self { Self { inner: BTreeMap::new() } }

    fn insert(&mut self, p: Paper) -> bool {
        let key = canonical_doi(&p.doi);
        if self.inner.contains_key(&key) { return false; }
        self.inner.insert(key, p);
        true
    }

    fn len(&self) -> usize { self.inner.len() }

    fn papers(&self) -> impl Iterator<Item = &Paper> {
        self.inner.values()
    }
}

fn canonical_doi(raw: &str) -> String {
    // Strip URL prefix, lowercase, trim
    raw.trim()
       .trim_start_matches("https://doi.org/")
       .trim_start_matches("http://dx.doi.org/")
       .to_lowercase()
       .to_string()
}

// ── SQLite setup ──────────────────────────────────────────────────────────────
fn open_db(path: &str) -> Result<Connection> {
    let conn = Connection::open(path)
        .with_context(|| format!("Cannot open {path}"))?;

    conn.execute_batch("
        CREATE TABLE IF NOT EXISTS papers (
            doi      TEXT PRIMARY KEY,
            title    TEXT,
            authors  TEXT,
            year     INTEGER,
            abstract TEXT,
            source   TEXT,
            url      TEXT
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts
        USING fts5(title, authors, abstract, content='papers', content_rowid='rowid');
    ")?;
    Ok(conn)
}

fn upsert(conn: &Connection, p: &Paper) -> Result<bool> {
    let rows = conn.execute(
        "INSERT OR IGNORE INTO papers (doi,title,authors,year,abstract,source,url)
         VALUES (?1,?2,?3,?4,?5,?6,?7)",
        params![p.doi, p.title, p.authors, p.year, p.abstract_, p.source, p.url],
    )?;
    if rows > 0 {
        conn.execute(
            "INSERT INTO papers_fts (title,authors,abstract)
             VALUES (?1,?2,?3)",
            params![p.title, p.authors, p.abstract_],
        )?;
    }
    Ok(rows > 0)
}

// ── Semantic Scholar ──────────────────────────────────────────────────────────
#[derive(Deserialize, Debug)]
struct S2Response { data: Vec<S2Paper> }

#[derive(Deserialize, Debug)]
struct S2Paper {
    #[serde(rename = "externalIds")]
    external_ids: Option<serde_json::Value>,
    title:   Option<String>,
    authors: Option<Vec<S2Author>>,
    year:    Option<i32>,
    #[serde(rename = "abstract")]
    abstract_: Option<String>,
    url:     Option<String>,
}

#[derive(Deserialize, Debug)]
struct S2Author { name: String }

fn fetch_semantic_scholar(query: &str, limit: usize) -> Result<Vec<Paper>> {
    let url = format!(
        "https://api.semanticscholar.org/graph/v1/paper/search\
         ?query={}&limit={}&fields=title,authors,year,abstract,externalIds,url",
        urlencoding(query), limit
    );

    let client = reqwest::blocking::Client::builder()
        .user_agent("jalali-crawler/0.1 (academic; gabriel@ucla.edu)")
        .timeout(std::time::Duration::from_secs(15))
        .build()?;

    let resp: S2Response = client.get(&url).send()?.json()?;

    let papers = resp.data.into_iter().filter_map(|s| {
        let doi = s.external_ids.as_ref()
            .and_then(|e| e.get("DOI"))
            .and_then(|d| d.as_str())
            .unwrap_or("")
            .to_string();
        let title = s.title.unwrap_or_default();
        if title.is_empty() { return None; }
        Some(Paper {
            doi:      if doi.is_empty() { format!("s2:{}", &title[..title.len().min(40)]) } else { doi },
            title,
            authors:  s.authors.unwrap_or_default().iter().map(|a| a.name.clone()).collect::<Vec<_>>().join(", "),
            year:     s.year.unwrap_or(0),
            abstract_: s.abstract_.unwrap_or_default(),
            source:   "semantic_scholar".to_string(),
            url:      s.url.unwrap_or_default(),
        })
    }).collect();

    Ok(papers)
}

// ── arXiv ─────────────────────────────────────────────────────────────────────
fn fetch_arxiv(query: &str, limit: usize) -> Result<Vec<Paper>> {
    let url = format!(
        "http://export.arxiv.org/api/query?search_query=all:{}&max_results={}&sortBy=relevance",
        urlencoding(query), limit
    );

    let client = reqwest::blocking::Client::builder()
        .user_agent("jalali-crawler/0.1")
        .timeout(std::time::Duration::from_secs(20))
        .build()?;

    let body = client.get(&url).send()?.text()?;

    // Parse Atom XML with scraper (it handles XML too)
    let doc = Html::parse_document(&body);
    let entry_sel  = Selector::parse("entry").unwrap();
    let title_sel  = Selector::parse("title").unwrap();
    let author_sel = Selector::parse("author > name").unwrap();
    let summary_sel= Selector::parse("summary").unwrap();
    let id_sel     = Selector::parse("id").unwrap();
    let published_sel = Selector::parse("published").unwrap();
    let year_re    = Regex::new(r"(\d{4})").unwrap();

    let mut papers = Vec::new();
    for entry in doc.select(&entry_sel) {
        let title = entry.select(&title_sel).next()
            .map(|e| e.text().collect::<String>().trim().replace('\n', " "))
            .unwrap_or_default();
        let arxiv_id = entry.select(&id_sel).next()
            .map(|e| e.text().collect::<String>().trim().to_string())
            .unwrap_or_default();
        let authors: Vec<String> = entry.select(&author_sel)
            .map(|e| e.text().collect::<String>().trim().to_string())
            .collect();
        let abstract_ = entry.select(&summary_sel).next()
            .map(|e| e.text().collect::<String>().trim().replace('\n', " "))
            .unwrap_or_default();
        let year_str = entry.select(&published_sel).next()
            .map(|e| e.text().collect::<String>())
            .unwrap_or_default();
        let year: i32 = year_re.captures(&year_str)
            .and_then(|c| c.get(1))
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(0);

        if title.is_empty() { continue; }
        papers.push(Paper {
            doi:      arxiv_id.clone(),
            title,
            authors:  authors.join(", "),
            year,
            abstract_,
            source:   "arxiv".to_string(),
            url:      arxiv_id,
        });
    }
    Ok(papers)
}

// ── URL scraper (generic) ─────────────────────────────────────────────────────
fn scrape_url(url: &str) -> Result<Paper> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("Mozilla/5.0 (compatible; jalali-crawler/0.1; academic)")
        .timeout(std::time::Duration::from_secs(15))
        .build()?;

    let body = client.get(url).send()?.text()?;
    let doc  = Html::parse_document(&body);

    // Try common meta tags
    let og_title  = meta(&doc, "og:title").or_else(|| meta(&doc, "title"));
    let og_desc   = meta(&doc, "og:description")
        .or_else(|| meta(&doc, "description"))
        .or_else(|| meta(&doc, "DC.Description"));
    let og_doi    = meta(&doc, "citation_doi")
        .or_else(|| meta(&doc, "DC.Identifier"));

    let title   = og_title.unwrap_or_else(|| {
        doc.select(&Selector::parse("title").unwrap()).next()
            .map(|e| e.text().collect::<String>())
            .unwrap_or_else(|| url.to_string())
    });

    Ok(Paper {
        doi:      og_doi.unwrap_or_else(|| format!("url:{url}")),
        title:    title.trim().to_string(),
        authors:  meta(&doc, "citation_author").unwrap_or_default(),
        year:     0,
        abstract_: og_desc.unwrap_or_default(),
        source:   "url".to_string(),
        url:      url.to_string(),
    })
}

fn meta(doc: &Html, name: &str) -> Option<String> {
    let sel = Selector::parse(
        &format!("meta[name='{name}'], meta[property='{name}']")
    ).ok()?;
    doc.select(&sel).next()
        .and_then(|e| e.value().attr("content"))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn urlencoding(s: &str) -> String {
    s.replace(' ', "+")
     .replace('"', "%22")
     .replace('&', "%26")
}

// ── Main ─────────────────────────────────────────────────────────────────────
fn main() -> Result<()> {
    let args   = Args::parse();
    let conn   = open_db(&args.db)?;
    let mut tree = PaperTree::new();

    println!("{}", "━━━ Jalali Lab Photonics Crawler ━━━".bright_cyan().bold());
    println!("db: {}\n", args.db.bright_yellow());

    // ── Semantic Scholar ────────────────────────────────────────────────────
    if let Some(ref q) = args.query {
        print!("{} ", "Semantic Scholar:".bright_blue());
        match fetch_semantic_scholar(q, args.limit) {
            Ok(papers) => {
                println!("{} results", papers.len().to_string().bright_green());
                for p in papers { tree.insert(p); }
            }
            Err(e) => println!("{}: {e}", "ERROR".bright_red()),
        }
    }

    // ── arXiv ───────────────────────────────────────────────────────────────
    if let Some(ref q) = args.arxiv {
        print!("{} ", "arXiv:".bright_blue());
        match fetch_arxiv(q, args.limit) {
            Ok(papers) => {
                println!("{} results", papers.len().to_string().bright_green());
                for p in papers { tree.insert(p); }
            }
            Err(e) => println!("{}: {e}", "ERROR".bright_red()),
        }
    }

    // ── URL list ────────────────────────────────────────────────────────────
    if let Some(ref path) = args.urls {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Cannot read {path}"))?;
        for url in content.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
            print!("{} {} ... ", "URL:".bright_blue(), url.dimmed());
            match scrape_url(url) {
                Ok(p) => {
                    let new = tree.insert(p);
                    println!("{}", if new { "ok".bright_green() } else { "dup".bright_yellow() });
                }
                Err(e) => println!("{}: {e}", "FAIL".bright_red()),
            }
        }
    }

    // ── Write to SQLite (BST order = sorted by DOI) ─────────────────────────
    println!("\n{} {} unique papers → db", "Writing".bright_cyan(), tree.len());
    let mut new_count = 0usize;
    for p in tree.papers() {
        if upsert(&conn, p)? { new_count += 1; }
        if args.verbose {
            println!("\n  {} {}", "▶".bright_cyan(), p.title.bright_white());
            println!("    {}", p.authors.dimmed());
            if !p.abstract_.is_empty() {
                let snip = &p.abstract_[..p.abstract_.len().min(200)];
                println!("    {snip}...");
            }
        }
    }

    println!("{} {} new  {} duplicate",
        "Done.".bright_green().bold(),
        new_count.to_string().bright_green(),
        (tree.len() - new_count).to_string().bright_yellow()
    );

    // ── Print FTS5 sample query ──────────────────────────────────────────────
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM papers", [], |r| r.get(0))?;
    println!("\nTotal papers in db: {}", count.to_string().bright_cyan());
    println!("FTS5 query example:");
    println!("  {}", "SELECT title, snippet(papers_fts,2,'>>','<<','...',20)".bright_yellow());
    println!("  {}",  "FROM papers_fts WHERE papers_fts MATCH 'phase retrieval';".bright_yellow());

    Ok(())
}
