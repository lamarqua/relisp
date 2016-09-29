#![allow(dead_code)]
#![allow(unused_imports)]

extern crate regex;

use std::collections::HashMap;
use regex::{Regex, RegexBuilder};
use std::io;
use std::io::prelude::*;
use std::cell::Cell;
use std::rc::Rc;
use std::time::SystemTime;
// use std::Box;

macro_rules! log(
    ($($arg:tt)*) => { {
        let r = write!(&mut ::std::io::stderr(), $($arg)*);
        r.unwrap();
    } }
);

static DEBUG: bool = true;

macro_rules! debug(
    ($($arg:tt)*) => { {
        if DEBUG {
            // let now = SystemTime::now();
            let r = write!(&mut ::std::io::stderr(), "\x1b[91;40m[DEBUG]\x1b[33;0m ");
            r.unwrap();
            let r = writeln!(&mut ::std::io::stderr(), $($arg)*);
            r.unwrap();
        }
    } }
);

fn unimplemented<T>() -> Result<T, String> {
    let res: Result<T, String> = Err("Not implemented yet!".to_string());
    res
}

#[derive(Debug)]
enum Atom {
    Nil,
    Int(i64),
    Float(f64),
    // Symbol(String),
    Quote(Rc<Value>),
}

#[derive(Debug)]
struct ConsCell {
    car: Rc<Value>,
    cdr: Rc<Value>,
    // is_list: Cell<bool>, 
}

#[derive(Debug)]
enum Value {
    Atom(Atom),
    ConsCell(ConsCell),
}

#[derive(Clone, Copy, Debug)]
enum TokenType {
    Whitespace,
    Open,
    Close,
    Dot,
    Quote,
    Float,
    Int,
    Symbol,
    Eof,
}

struct Lexer<'a> {
    input_string: &'a str,
    last_read_token: Option<Result<(TokenType, &'a str, usize), String>>,
    full_input_string: &'a str,
    rules: Vec<(Regex, TokenType)>,
}

impl<'a> Lexer<'a> {
    fn new(input_string: &'a str) -> Self {
        fn build_regex(re: &str) -> Regex {
            let builder = RegexBuilder::new(re);
            return builder.case_insensitive(true)
                          .ignore_whitespace(true)
                          .compile()
                          .unwrap();
        }

        let rules_array = [
            (r"^\s+", TokenType::Whitespace),
            (r"^[\(\{\[]", TokenType::Open),
            (r"^[\)\}\]]", TokenType::Close),
            (r"^\.", TokenType::Dot),
            (r"^'", TokenType::Quote),
            (r"^[+-]?\d*\.\d+(?:[eE][-]\d+)?", TokenType::Float),
            (r"^[+-]?\d+", TokenType::Int),
            (r"^\w[\w\d_!?+-=/]*", TokenType::Symbol),
            (r"^$", TokenType::Eof)];
        let rules = rules_array.iter().map(move |x| (build_regex(x.0), x.1)).collect();
        Lexer {
            input_string: input_string,
            full_input_string: input_string,
            last_read_token: None,
            rules: rules,
        }
    }

    fn _find_next_token(&mut self) -> Result<(TokenType, &'a str, usize), String> {
        self.last_read_token.clone().unwrap_or_else(move || {
            'skip_whitespace: loop {
                for &(ref rule, token_type) in &self.rules {
                    if let Some((0, x)) = rule.find(self.input_string) {
                        let matched = &self.input_string[..x];
                        // debug!("{:?} {:?} {} {}", token_type, matched, self.input_string, x);
                        if let TokenType::Whitespace = token_type {
                            self.input_string = &self.input_string[x..];
                            continue 'skip_whitespace;
                        }
                        let res = Ok((token_type, matched, x));
                        self.last_read_token = Some(res.clone());
                        return res;
                    }
                }
                let err = Err(format!("Unexpected input: '{}'\n. Context {}", self.input_string, self.full_input_string));
                self.last_read_token = Some(err.clone());
                return err;
            }
        })
    }

    fn peek_token(&mut self) -> Result<(TokenType, &'a str), String> {
        self._find_next_token().map(|(tt, m, _)| (tt, m))
    }

    fn next_token(&mut self) -> Result<(TokenType, &'a str), String> {
        self._find_next_token().map(|(tt, m, x)| {
            self.input_string = &self.input_string[x..];
            self.last_read_token = None;
            return (tt, m);
        })
    }
}

struct Environment<'a> {
    table: HashMap<String, Value>,
    parent: Option<&'a Environment<'a>>
}

impl<'a> Environment<'a> {
    fn new(parent: Option<&'a Environment>) -> Self {
        Environment {
            table: HashMap::new(),
            parent: parent,
        }
    }
}

fn parse(input_string: &str) -> Result<Vec<Value>, String> {
    let mut lexer = Lexer::new(input_string);

    fn parens_match(o: &str, c: &str) -> bool {
        o == "(" && c == ")" ||
        o == "[" && c == "]" ||
        o == "{" && c == "}" 
    }

    fn parse_value(mut lexer: &mut Lexer) -> Result<Value, String> {
        let (token_type, matched) = try!(lexer.next_token());

        debug!("parse_value: {:?} {:?}: ", token_type, matched);
        match token_type {
            TokenType::Int => {
                let v = Value::Atom(Atom::Int(matched.parse::<i64>().unwrap()));
                return Ok(v);
            }
            TokenType::Float => {
                let v = Value::Atom(Atom::Float(matched.parse::<f64>().unwrap()));
                return Ok(v);
            }
            TokenType::Open => {
                let opening_paren = matched;

                let cell = try!(parse_cell(&mut lexer));

                let (_, closing_paren) = try!(lexer.next_token());
                if !parens_match(opening_paren, closing_paren) {
                    return Err(format!("Non-matching paren!: {} {}", opening_paren, closing_paren));
                } else {
                    return Ok(Value::ConsCell(cell));
                }
            }
            TokenType::Quote => {
                unreachable!();
            }
            TokenType::Whitespace => {
                unreachable!();
            }
            _ => {
                Err(format!("Unexpected token {:?} {}", token_type, matched))
            }
        }
    }

    fn parse_cell(lexer: &mut Lexer) -> Result<ConsCell, String> {
        let value = try!(parse_value(lexer));

        let (token_type, matched) = try!(lexer.peek_token());
        if let TokenType::Dot = token_type {
        } 

        unimplemented()
    }

    // Lexer test
    // TODO: make it a proper test
    // loop {
    //     let r = lexer.next_token();
    //     match r {
    //         Ok((tt, n)) => {
    //             debug!("{:?} {}", tt, n);
    //             if let TokenType::Eof = tt {
    //                 break;
    //             }
    //         }
    //         Err(s) => {
    //             debug!("Err in lexer() {}", s);
    //             break;
    //         }
    //     }
    // }

    let mut res = Vec::new();
    loop {
        let value = try!(parse_value(&mut lexer));
        res.push(value);

        // TODO parse values

        break;
    }

    return Ok(Vec::new());
}

fn print_help() {
    log!(":load (or :l) to load an external file\n");
    log!(":exit to quit\n");
}
 
fn main() {
    log!("(R)ust (E)xperiments in (LISP)\n");
    debug!("DEBUG activated");
    loop {
        log!("λ ");

        let mut raw_input_line = String::new();
        
        match io::stdin().read_line(&mut raw_input_line) {
            Err(error) => { log!("error: {}", error) }
            Ok(_) => (),
        }
        
        let input_line: &str = raw_input_line.trim();

        match input_line {
            "" | ":exit" => { log!("Goodbye."); break; }
            ":help" => { print_help(); }
            _ => {
                let exprs = parse(input_line);
                // let results = eval(exprs);
                // print(results);
            }
        }
    }
}
