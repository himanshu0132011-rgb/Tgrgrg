package com.byteai.local;

import android.content.Context;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Offline local agent: deterministic tool routing plus BYTE's small text model. */
public final class ByteAgent {
    private final Context appContext;
    private final ByteEngine engine;
    private final File memoryFile;

    public ByteAgent(Context context, ByteEngine engine) {
        this.appContext = context.getApplicationContext();
        this.engine = engine;
        this.memoryFile = new File(appContext.getFilesDir(), "byte_memory.txt");
    }

    public String respond(String input) {
        String text = input == null ? "" : input.trim();
        String lower = text.toLowerCase(Locale.ROOT);
        if (text.isEmpty()) return "Type a message. Try /help to see my local tools.";

        if (lower.equals("/help") || lower.equals("help") || lower.equals("what can you do") || lower.equals("what can you do?")) {
            return "BYTE local agent tools\n\n" +
                    "• remember <note> — save a note on this phone\n" +
                    "• recall — list saved notes\n" +
                    "• search memory for <words> — search saved notes\n" +
                    "• calculate <expression> — basic arithmetic\n" +
                    "• status — show the local model details\n" +
                    "• generate <prompt> — sample from the tiny neural model\n\n" +
                    "Other messages are passed to the experimental text predictor. It is not reliable general-purpose Q&A.";
        }
        if (lower.equals("/status") || lower.equals("status")) {
            return "BYTE v0.2 · LOCAL / OFFLINE\n" +
                    "Vocabulary: " + engine.vocabularySize() + " tokens\n" +
                    "Context window: " + engine.contextSize() + " tokens\n" +
                    "Embedding size: " + engine.embeddingSize() + "\n" +
                    "Hidden layer: " + engine.hiddenSize() + " units\n" +
                    "Saved notes: " + readNotes().size() + "\n" +
                    "Pretrained model: no\nInternet/API required: no";
        }

        Matcher remember = Pattern.compile("^(?:/remember|remember|save(?: this)?)(?:\\s*[:, -]?\\s+)(.+)$", Pattern.CASE_INSENSITIVE | Pattern.DOTALL).matcher(text);
        if (remember.matches()) return saveNote(remember.group(1));

        if (lower.equals("/recall") || lower.equals("recall") || lower.equals("what do you remember") ||
                lower.equals("what do you remember?") || lower.equals("show my memory")) {
            return listNotes("");
        }
        Matcher search = Pattern.compile("^(?:search memory for|search my notes for)\\s+(.+)$", Pattern.CASE_INSENSITIVE).matcher(text);
        if (search.matches()) return listNotes(search.group(1));

        Matcher calc = Pattern.compile("^(?:/calculate|calculate|what is)\\s+(.+?)[?]?$", Pattern.CASE_INSENSITIVE).matcher(text);
        if (calc.matches() && calc.group(1).matches("[0-9\\s()+\\-*/%.]+")) {
            try {
                double result = new ExpressionParser(calc.group(1)).parse();
                return calc.group(1).trim() + " = " + (result == Math.rint(result) ? Long.toString((long) result) : Double.toString(result));
            } catch (RuntimeException ex) {
                return "Calculator error: " + ex.getMessage();
            }
        }

        Matcher generate = Pattern.compile("^(?:/generate|generate)\\s+(.+)$", Pattern.CASE_INSENSITIVE | Pattern.DOTALL).matcher(text);
        if (generate.matches()) return "Experimental continuation (not a verified answer):\n" + engine.generate(generate.group(1), 36);

        return "I can run local tools, but this tiny model cannot reliably answer general questions yet.\n\n" +
                "Raw model continuation (may be incoherent):\n" + engine.generate(text, 32);
    }

    private String saveNote(String note) {
        String clean = note.trim().replace('\n', ' ');
        if (clean.isEmpty()) return "Tell me what to remember. Example: remember my project is called BYTE.";
        try (FileOutputStream out = new FileOutputStream(memoryFile, true)) {
            out.write((clean + "\n").getBytes(StandardCharsets.UTF_8));
            return "Saved to BYTE's private local memory on this phone. This stores the note; it does not retrain the model.";
        } catch (IOException ex) {
            return "I couldn't save that note: " + ex.getMessage();
        }
    }

    private List<String> readNotes() {
        List<String> result = new ArrayList<>();
        if (!memoryFile.exists()) return result;
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(memoryFile), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) if (!line.trim().isEmpty()) result.add(line.trim());
        } catch (IOException ignored) { }
        return result;
    }

    private String listNotes(String query) {
        List<String> notes = readNotes();
        if (!query.isEmpty()) {
            String[] terms = query.toLowerCase(Locale.ROOT).split("\\s+");
            List<String> filtered = new ArrayList<>();
            for (String note : notes) {
                String normalized = note.toLowerCase(Locale.ROOT);
                boolean matches = true;
                for (String term : terms) if (!term.isEmpty() && !normalized.contains(term)) matches = false;
                if (matches) filtered.add(note);
            }
            notes = filtered;
        }
        if (notes.isEmpty()) return query.isEmpty() ? "BYTE's memory is empty. Try: remember <note>." : "No matching note found in BYTE's local memory.";
        StringBuilder out = new StringBuilder("BYTE local memory\n");
        int start = Math.max(0, notes.size() - 20);
        for (int i = start; i < notes.size(); i++) out.append(i - start + 1).append(". ").append(notes.get(i)).append('\n');
        return out.toString().trim();
    }

    /** Recursive-descent parser. Only numeric arithmetic is accepted; never evaluates Java or scripts. */
    private static final class ExpressionParser {
        private final String s;
        private int pos = 0;
        ExpressionParser(String s) { this.s = s; }
        double parse() {
            double value = expression();
            spaces();
            if (pos != s.length()) throw new IllegalArgumentException("Unexpected character at position " + pos + ".");
            if (Double.isNaN(value) || Double.isInfinite(value)) throw new IllegalArgumentException("Result is not finite.");
            return value;
        }
        private double expression() {
            double value = term();
            while (true) {
                spaces();
                if (take('+')) value += term();
                else if (take('-')) value -= term();
                else return value;
            }
        }
        private double term() {
            double value = factor();
            while (true) {
                spaces();
                if (take('*')) value *= factor();
                else if (take('/')) { double divisor = factor(); if (divisor == 0) throw new IllegalArgumentException("Division by zero."); value /= divisor; }
                else if (take('%')) { double divisor = factor(); if (divisor == 0) throw new IllegalArgumentException("Division by zero."); value %= divisor; }
                else return value;
            }
        }
        private double factor() {
            spaces();
            if (take('+')) return factor();
            if (take('-')) return -factor();
            if (take('(')) { double value = expression(); spaces(); if (!take(')')) throw new IllegalArgumentException("Missing closing parenthesis."); return value; }
            int start = pos;
            while (pos < s.length() && (Character.isDigit(s.charAt(pos)) || s.charAt(pos) == '.')) pos++;
            if (start == pos) throw new IllegalArgumentException("Expected a number at position " + pos + ".");
            try { return Double.parseDouble(s.substring(start, pos)); }
            catch (NumberFormatException ex) { throw new IllegalArgumentException("Invalid number."); }
        }
        private boolean take(char c) { if (pos < s.length() && s.charAt(pos) == c) { pos++; return true; } return false; }
        private void spaces() { while (pos < s.length() && Character.isWhitespace(s.charAt(pos))) pos++; }
    }
}
