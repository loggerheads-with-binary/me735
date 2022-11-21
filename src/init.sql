CREATE TABLE IF NOT EXISTS entries(

    tstamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
    holder_code INTEGER, 
    src TEXT NOT NULL,
    no_of_operations INTEGER DEFAULT 0,
    operations TEXT DEFAULT "{}",
    stability_1 REAL DEFAULT 0,
    time_taken REAL DEFAULT 0, 
    fpath TEXT DEFAULT ""

); 