/* Reader for the YAML subset that PyYAML's safe_dump writes for a run's config.yaml (block mappings and
   sequences, plain / single-quoted / double-quoted scalars folded over several lines, [] and {}).
   Anything outside that subset throws, and the caller then shows the verbatim file instead of a parsed view. */

export type Yaml = string | number | boolean | null | Yaml[] | { [k: string]: Yaml };

interface Line { indent: number; text: string }

export class YamlSubsetError extends Error {}

const KEY = /^([A-Za-z_][\w.-]*):(?:\s+(.*))?$/;
const isItem = (t: string) => t === "-" || t.startsWith("- ");

export function parseYaml(src: string): Yaml {
  const raws = src.split(/\r?\n/).filter((raw) => raw.trim() !== "");
  if (raws.some((raw) => /^ *\t/.test(raw))) throw new YamlSubsetError("tab indentation");
  const lines: Line[] = raws.map((raw) => ({ indent: raw.length - raw.trimStart().length, text: raw.trim() }));
  let i = 0;
  const skipComments = () => { while (i < lines.length && lines[i]!.text.startsWith("#")) i++; };

  function block(): Yaml {
    skipComments();
    const l = lines[i]!;
    return isItem(l.text) ? seq(l.indent) : map(l.indent);
  }

  function map(ind: number): Yaml {
    const out: { [k: string]: Yaml } = {};
    for (skipComments(); i < lines.length && lines[i]!.indent === ind && !isItem(lines[i]!.text); skipComments()) {
      const m = KEY.exec(lines[i]!.text);
      if (!m) throw new YamlSubsetError(`unsupported line: ${lines[i]!.text}`);
      i++;
      const rest = m[2];
      if (rest === undefined || rest === "") {
        const nx = lines[i];
        out[m[1]!] = nx && (nx.indent > ind || (nx.indent === ind && isItem(nx.text))) ? block() : null;
      } else {
        out[m[1]!] = scalar(rest, ind);
      }
    }
    if (i < lines.length && lines[i]!.indent > ind) throw new YamlSubsetError(`unexpected indentation: ${lines[i]!.text}`);
    return out;
  }

  function seq(ind: number): Yaml {
    const out: Yaml[] = [];
    for (skipComments(); i < lines.length && lines[i]!.indent === ind && isItem(lines[i]!.text); skipComments()) {
      const rest = lines[i]!.text.slice(1).trimStart();
      if (rest === "") { i++; out.push(block()); continue; }
      if (KEY.test(rest)) {            // "- key: value" starts a mapping item
        lines[i] = { indent: ind + 2, text: rest };
        out.push(map(ind + 2));
        continue;
      }
      i++;
      out.push(scalar(rest, ind));
    }
    return out;
  }

  /** A scalar that starts on the current line; continuation lines are those indented deeper than `parent`. */
  function scalar(first: string, parent: number): Yaml {
    const q = first[0];
    if (q === "'" || q === '"') {
      let text = first;
      while (!closed(text, q) && i < lines.length && lines[i]!.indent > parent) text += ` ${lines[i++]!.text}`;
      if (!closed(text, q)) throw new YamlSubsetError(`unterminated quoted scalar: ${first}`);
      const body = text.slice(1, text.length - 1);
      return q === "'" ? body.replace(/''/g, "'") : unescapeDouble(body);
    }
    let text = first;
    while (i < lines.length && lines[i]!.indent > parent && !lines[i]!.text.startsWith("#")) text += ` ${lines[i++]!.text}`;
    return plain(text);
  }

  const out = lines.length ? block() : null;
  skipComments();
  if (i < lines.length) throw new YamlSubsetError(`unparsed content: ${lines[i]!.text}`);
  return out;
}

function closed(text: string, q: string): boolean {
  if (text.length < 2 || text[text.length - 1] !== q) return false;
  const body = text.slice(1, -1);
  if (q === "'") return body.replace(/''/g, "").indexOf("'") === -1;          // '' is an escaped quote
  let trailing = 0;
  for (let j = body.length - 1; j >= 0 && body[j] === "\\"; j--) trailing++;
  return trailing % 2 === 0 && body.replace(/\\./g, "").indexOf('"') === -1;  // the final quote is not escaped
}

function unescapeDouble(s: string): string {
  return s.replace(/\\(["\\/nt0]|x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4})/g, (_, e: string) => {
    if (e === "n") return "\n";
    if (e === "t") return "\t";
    if (e === "0") return "\0";
    if (e[0] === "x" || e[0] === "u") return String.fromCharCode(parseInt(e.slice(1), 16));
    return e;
  });
}

function plain(t: string): Yaml {
  if (t === "null" || t === "~" || t === "Null" || t === "NULL") return null;
  if (t === "true" || t === "True") return true;
  if (t === "false" || t === "False") return false;
  if (t === "[]") return [];
  if (t === "{}") return {};
  if (/^[-+]?\d+$/.test(t)) return Number(t);
  if (/^[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?$/.test(t)) return Number(t);
  if (t === ".nan" || t === ".NaN") return null;          // undefined, never 0
  if (/^[-+]?\.inf$/i.test(t)) return null;
  if (/^[[{]/.test(t)) throw new YamlSubsetError(`flow collections are not supported: ${t}`);
  return t;
}
