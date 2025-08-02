-- Migration to update wake_words table to support multi-word phrases and fuzzy matching

-- Step 1: Drop the old unique constraint
ALTER TABLE wake_words DROP CONSTRAINT IF EXISTS uq_org_wake_word;

-- Step 2: Rename the word column to phrase
ALTER TABLE wake_words RENAME COLUMN word TO phrase;

-- Step 3: Increase the size of the phrase column
ALTER TABLE wake_words ALTER COLUMN phrase TYPE VARCHAR(500);

-- Step 4: Add new columns for fuzzy matching configuration
ALTER TABLE wake_words ADD COLUMN IF NOT EXISTS max_edit_distance INTEGER DEFAULT 2 NOT NULL;
ALTER TABLE wake_words ADD COLUMN IF NOT EXISTS similarity_threshold FLOAT DEFAULT 0.8 NOT NULL;

-- Step 5: Add new columns for clip configuration
ALTER TABLE wake_words ADD COLUMN IF NOT EXISTS pre_roll_seconds INTEGER DEFAULT 10 NOT NULL;
ALTER TABLE wake_words ADD COLUMN IF NOT EXISTS post_roll_seconds INTEGER DEFAULT 30 NOT NULL;

-- Step 6: Add the new unique constraint
ALTER TABLE wake_words ADD CONSTRAINT uq_org_wake_phrase UNIQUE (organization_id, phrase);