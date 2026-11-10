package com.byteai.local;

import android.content.Context;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Pure Java inference port of BYTE's tiny NumPy MLP. No network or pretrained model. */
public final class ByteEngine {
    private static final Pattern TOKEN_PATTERN = Pattern.compile("[\\p{L}\\p{N}_]+|[^\\s\\p{L}\\p{N}_]");
    private static final String[] SPECIAL = {"<PAD>", "<BOS>", "<EOS>", "<UNK>"};
    private final List<String> tokens = new ArrayList<>();
    private final java.util.Map<String, Integer> tokenIds = new java.util.HashMap<>();
    private float[][] embeddings;
    private float[][] w1;
    private float[] b1;
    private float[][] w2;
    private float[] b2;
    private int vocabSize, contextSize, embeddingSize, hiddenSize;

    public ByteEngine(Context context) throws Exception {
        loadTokenizer(context);
        loadWeights(context);
        if (tokens.size() != vocabSize) throw new IOException("Tokenizer/model vocabulary mismatch.");
    }

    public int vocabularySize() { return vocabSize; }
    public int contextSize() { return contextSize; }
    public int embeddingSize() { return embeddingSize; }
    public int hiddenSize() { return hiddenSize; }

    private void loadTokenizer(Context context) throws Exception {
        byte[] bytes;
        try (java.io.InputStream in = context.getAssets().open("tokenizer.json")) {
            bytes = readAll(in);
        }
        JSONObject json = new JSONObject(new String(bytes, java.nio.charset.StandardCharsets.UTF_8));
        JSONArray array = json.getJSONArray("tokens");
        for (int i = 0; i < array.length(); i++) {
            String token = array.getString(i);
            tokens.add(token);
            tokenIds.put(token, i);
        }
        if (tokens.size() < 5) throw new IOException("Invalid tokenizer vocabulary.");
        for (int i = 0; i < SPECIAL.length; i++) {
            if (!SPECIAL[i].equals(tokens.get(i))) throw new IOException("Invalid special-token order.");
        }
    }

    private static byte[] readAll(java.io.InputStream in) throws IOException {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        int count;
        while ((count = in.read(buffer)) != -1) out.write(buffer, 0, count);
        return out.toByteArray();
    }

    private void loadWeights(Context context) throws Exception {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(context.getAssets().open("byte_weights.bin")))) {
            byte[] magic = new byte[4];
            in.readFully(magic);
            if (!new String(magic, java.nio.charset.StandardCharsets.US_ASCII).equals("BYTE")) throw new IOException("Invalid model header.");
            int version = in.readInt();
            if (version != 1) throw new IOException("Unsupported model version: " + version);
            vocabSize = in.readInt();
            contextSize = in.readInt();
            embeddingSize = in.readInt();
            hiddenSize = in.readInt();
            if (vocabSize != tokens.size() || contextSize < 1 || embeddingSize < 1 || hiddenSize < 1) throw new IOException("Invalid model dimensions.");
            embeddings = readMatrix(in, vocabSize, embeddingSize);
            w1 = readMatrix(in, contextSize * embeddingSize, hiddenSize);
            b1 = readVector(in, hiddenSize);
            w2 = readMatrix(in, hiddenSize, vocabSize);
            b2 = readVector(in, vocabSize);
            if (in.read() != -1) throw new IOException("Unexpected bytes after model weights.");
        }
    }

    private static float[][] readMatrix(DataInputStream in, int rows, int cols) throws IOException {
        float[][] result = new float[rows][cols];
        for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) result[r][c] = in.readFloat();
        return result;
    }
    private static float[] readVector(DataInputStream in, int size) throws IOException {
        float[] result = new float[size];
        for (int i = 0; i < size; i++) result[i] = in.readFloat();
        return result;
    }

    public List<Integer> encode(String text) {
        List<Integer> ids = new ArrayList<>();
        Matcher matcher = TOKEN_PATTERN.matcher(text);
        while (matcher.find()) ids.add(tokenIds.getOrDefault(matcher.group().toLowerCase(Locale.ROOT), 3));
        return ids;
    }

    public String decode(List<Integer> ids) {
        StringBuilder out = new StringBuilder();
        for (int id : ids) {
            if (id < 0 || id >= tokens.size()) id = 3;
            String piece = tokens.get(id);
            if (isSpecial(piece)) continue;
            if (out.length() == 0) out.append(piece);
            else if (",.!?;:)]}".contains(piece) && piece.length() == 1) out.append(piece);
            else if (out.length() > 0 && "([{".indexOf(out.charAt(out.length() - 1)) >= 0) out.append(piece);
            else out.append(' ').append(piece);
        }
        return out.toString();
    }

    private static boolean isSpecial(String token) {
        for (String special : SPECIAL) if (special.equals(token)) return true;
        return false;
    }

    private float[] logitsFor(List<Integer> history) {
        int[] context = new int[contextSize];
        int from = Math.max(0, history.size() - contextSize);
        int used = history.size() - from;
        for (int i = 0; i < contextSize - used; i++) context[i] = 1;
        for (int i = 0; i < used; i++) context[contextSize - used + i] = history.get(from + i);
        float[] flat = new float[contextSize * embeddingSize];
        int at = 0;
        for (int id : context) {
            if (id < 0 || id >= vocabSize) id = 3;
            for (int j = 0; j < embeddingSize; j++) flat[at++] = embeddings[id][j];
        }
        float[] hidden = new float[hiddenSize];
        for (int h = 0; h < hiddenSize; h++) {
            float sum = b1[h];
            for (int i = 0; i < flat.length; i++) sum += flat[i] * w1[i][h];
            hidden[h] = (float)Math.tanh(sum);
        }
        float[] logits = new float[vocabSize];
        for (int v = 0; v < vocabSize; v++) {
            float sum = b2[v];
            for (int h = 0; h < hiddenSize; h++) sum += hidden[h] * w2[h][v];
            logits[v] = sum;
        }
        return logits;
    }

    public String generate(String prompt, int maxNewTokens) {
        List<Integer> generated = encode(prompt);
        List<Integer> history = new ArrayList<>();
        for (int i = 0; i < contextSize; i++) history.add(1);
        history.addAll(generated);
        Random random = new Random();
        for (int step = 0; step < maxNewTokens; step++) {
            float[] logits = logitsFor(history);
            // Keep the 12 highest logits, matching the desktop experiment's top-k sampling.
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < vocabSize; i++) indices.add(i);
            indices.sort((a, b) -> Float.compare(logits[b], logits[a]));
            int k = Math.min(12, vocabSize);
            double max = logits[indices.get(0)] / 0.85;
            double sum = 0;
            double[] probs = new double[k];
            for (int i = 0; i < k; i++) {
                probs[i] = Math.exp(logits[indices.get(i)] / 0.85 - max);
                sum += probs[i];
            }
            double pick = random.nextDouble() * sum;
            int chosen = indices.get(k - 1);
            for (int i = 0; i < k; i++) {
                pick -= probs[i];
                if (pick <= 0) { chosen = indices.get(i); break; }
            }
            if (chosen == 2) break;
            generated.add(chosen);
            history.add(chosen);
        }
        return decode(generated);
    }
}
