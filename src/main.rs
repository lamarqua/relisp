#![allow(unused_variables)]
#![allow(dead_code)]

extern crate regex;

use regex::{Regex, RegexBuilder};
use std::collections::HashMap;
use std::fmt::Debug;
use std::io::prelude::*;
use std::io;
use std::rc::Rc;

macro_rules! log(
    ($($arg:tt)*) => { {
        let r = write!(&mut ::std::io::stderr(), $($arg)*);
        r.unwrap();
    } }
);

static DEBUG: bool = true;
// const MAX_STACK_DEPTH: usize = 512;

macro_rules! debug(
    ($($arg:tt)*) => { {
        if DEBUG {
            // TODO: add timestamp
            let r = write!(&mut ::std::io::stderr(), "\x1b[91;40m[DEBUG]\x1b[33;0m ");
            r.unwrap();
            let r = writeln!(&mut ::std::io::stderr(), $($arg)*);
            r.unwrap();
        }
    } }
);

fn debug_dump<S: Debug>(s: S) {
    debug!("{:?}", s);
}

fn unimplemented<T>() -> Result<T, String> {
    let res: Result<T, String> = Err("Not implemented yet!".to_string());
    res
}

#[derive(Debug, Clone)]
enum Atom {
    Int(i64),
    Float(f64),
    String(String),
}

#[derive(Debug, Clone)]
enum ConsCell {
    Nil,
    Pair(Box<Value>, Box<Value>),
}

fn into_vec(list: &ConsCell) -> Option<Vec<Value>> {
    let mut res: Vec<Value> = Vec::new();

    let mut head: &ConsCell = list;
    loop {
        match *head {
            ConsCell::Nil => {
                break;
            }
            ConsCell::Pair(ref car, ref cdr_value) => {
                if let Value::ConsCell(ref cdr_cell) = **cdr_value {
                    res.push(*car.clone());
                    head = cdr_cell;
                } else {
                    return None;
                }
            }
        }
    }


    return Some(res);
}

// impl IntoIterator for ConsCell {
//     type Item = ConsCell;

// }

struct NativeProc(fn(&ConsCell, &Rc<Environment>) -> Value);

#[derive(Debug, Clone)]
enum ProcedureCode {
    InterpretredProc(Box<Value>),
    NativeProc(NativeProc),
}

impl Debug for NativeProc {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        // let NativeProc(fptr) = *self;
        write!(f, "<Native function pointer>")
    }
}

impl Clone for NativeProc {
    fn clone(&self) -> Self {
        let NativeProc(fptr) = *self;
        NativeProc(fptr)
    }
}

#[derive(Debug, Clone)]
enum Value {
    Atom(Atom),
    ConsCell(ConsCell),
    Symbol(String),
    Quote(Box<Value>),
    Procedure(ConsCell, ProcedureCode, Rc<Environment>),
    SpecialForm(String),
}   

#[derive(Clone, Debug, PartialEq)]
enum TokenType {
    Close,
    Comment,
    Dot,
    Eof,
    Float,
    Int,
    Open,
    Quote,
    String,
    Keyword,
    Symbol,
    Whitespace,
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
            (r#"^\s+"#, TokenType::Whitespace),
            (r#"^;[^\n]*(?:\n|$)"#, TokenType::Comment),
            (r#"^[\(\{\[]"#, TokenType::Open),
            (r#"^[\)\}\]]"#, TokenType::Close),
            (r#"^\."#, TokenType::Dot),
            (r#"^'"#, TokenType::Quote),
            (r#"^[+-]?\d*\.\d+(?:[eE][-]\d+)?"#, TokenType::Float),
            (r#"^[+-]?\d+"#, TokenType::Int),
            (r#""(?:[^\\"]|\\.)*""#, TokenType::String),
            (r#"^lambda|if|quote|define"#, TokenType::Keyword),
            (r#"^[\w\d_!?+-=/<>*]+"#, TokenType::Symbol),
            (r#"^$"#, TokenType::Eof)];
        let rules = rules_array.into_iter().map(move |x| (build_regex(x.0), x.1.clone())).collect();
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
                for &(ref rule, ref token_type) in &self.rules {
                    if let Some((0, x)) = rule.find(self.input_string) {
                        let matched = &self.input_string[..x];
                        // debug!("{:?} {:?} {} {}", token_type, matched, self.input_string, x);
                        match *token_type {
                            TokenType::Whitespace | TokenType::Comment => {
                                self.input_string = &self.input_string[x..];
                                continue 'skip_whitespace;
                            }
                            _ => ()
                        }
                        let res = Ok((token_type.clone(), matched, x));
                        self.last_read_token = Some(res.clone());
                        return res;
                    }
                }
                let err = Err(format!("Unexpected input: {}. Context {}", self.input_string, self.full_input_string));
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

    fn consume_token(&mut self) -> Result<(), String> {
        self.next_token().map(|_| ())
    }
}

fn eof_error<T>(context: &str) -> Result<T, String> {
    Err(format!("Unexpected EOF {}", context))
}

#[derive(Debug)]
struct Environment {
    table: HashMap<String, Value>,
    parent: Option<Rc<Environment>>
}

impl Environment {
    fn new(parent: Option<Rc<Environment>>) -> Self {
        Environment {
            table: HashMap::new(),
            parent: parent,
        }
    }

    fn insert(&mut self, symbol: &str, value: Value) {
        self.table.insert(symbol.to_string(), value);
    }

    fn lookup(&self, symbol: &str) -> Option<&Value> {
        if self.table.contains_key(symbol) {
            self.table.get(symbol).clone()
        } else {
            self.parent.as_ref().and_then(|parent| {
                parent.lookup(symbol)
            })
        }
    }
}

fn parse(input_string: &str) -> Result<Vec<Value>, String> {
    let mut lexer = Lexer::new(input_string);

    fn parens_match(o: &str, c: &str) -> bool {
        o == "(" && c == ")" ||
        o == "[" && c == "]" ||
        o == "{" && c == "}" // maybe replace with small LUT?
    }

    fn expect(lexer: &mut Lexer, expected_token_type: TokenType) -> Result<(), String> {
        let (token_type, matched) = try!(lexer.peek_token());
        if token_type != expected_token_type {
            Err(format!("Unexpected token {} (type {:?}). Expected {:?}",
                        matched, token_type, expected_token_type))
        } else {   
            Ok(())
        }
    }

    fn parse_value(lexer: &mut Lexer) -> Result<Value, String> {
        let (token_type, matched) = try!(lexer.next_token());

        // debug!("parse_value: {:?} {:?}: ", token_type, matched);
        match token_type { // we also parse the atoms directly
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

                let cell = try!(parse_cell(lexer));

                let (_, closing_paren) = try!(lexer.next_token());
                if !parens_match(opening_paren, closing_paren) {
                    return Err(format!("Non-matching paren! {} {}", opening_paren, closing_paren));
                } else {
                    return Ok(Value::ConsCell(cell));
                }
            }
            TokenType::String => {
                let v = Value::Atom(Atom::String(matched[1..matched.len()-1].to_string()));
                Ok(v)
            }
            TokenType::Symbol => {
                let v = Value::Symbol(matched.to_string());
                Ok(v)
            }
            TokenType::Quote => {
                let v = try!(parse_value(lexer));
                Ok(Value::Quote(Box::new(v)))
            }
            TokenType::Keyword => {
                let v = Value::SpecialForm(matched.to_string());
                Ok(v)
            }
            TokenType::Eof => eof_error(""),
            _ => {
                Err(format!("Unexpected token {:?} {}", token_type, matched))
            }
        }
    }

    fn parse_cell(lexer: &mut Lexer) -> Result<ConsCell, String> {
        let (token_type, _) = try!(lexer.peek_token());

        if let TokenType::Close = token_type {
            return Ok(ConsCell::Nil);
        }

        let car = Box::new(try!(parse_value(lexer)));
        let (token_type, _) = try!(lexer.peek_token());

        let cdr_value = match token_type {
            TokenType::Dot => {
                try!(lexer.consume_token());
                let cdr_value = try!(parse_value(lexer));
                try!(expect(lexer, TokenType::Close));
                cdr_value
            }
            _ => {
                let cell = try!(parse_cell(lexer));
                Value::ConsCell(cell)
            }
        };

        let cdr = Box::new(cdr_value);
        Ok(ConsCell::Pair(car, cdr))
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
        let (token_type, _) = try!(lexer.peek_token());
        if let TokenType::Eof = token_type {
            break;
        }
        let value = try!(parse_value(&mut lexer));
        res.push(value);
    }

    return Ok(res);
}

fn eval(value: &Value, env: Rc<Environment>) -> Result<Value, String> {
    fn call_proc(env: &Rc<Environment>, arg_names: &ConsCell, arg_values: &ConsCell, body: &ProcedureCode) -> Result<Value, String> {

        let proc_env = Environment::new(Some(env.clone()));

        match *body {
            ProcedureCode::InterpretredProc(_) => {
                let mut cur_arg_name = arg_names;
                loop {
                    match *cur_arg_name {
                        ConsCell::Nil => {
                            
                        }
                        ConsCell::Pair(ref car, ref cdr) => {

                        }
                    }
                }
            }
            ProcedureCode::NativeProc(NativeProc(ref fun)) => {
                let result = fun(arg_values, &env);

            }
        }

        unimplemented()
    }

    match *value {
        Value::Atom(_) => Ok(value.clone()),
        Value::Quote(ref quoted_val) => Ok(*quoted_val.clone()),
        Value::Procedure(_, _, _) => Ok(value.clone()),
        Value::ConsCell(ConsCell::Nil) => Ok(value.clone()),
        Value::Symbol(ref symbol_name) => env.lookup(&symbol_name)
                                        .ok_or(format!("Unknown symbol {}", &symbol_name))
                                        .map(|x| x.clone()),
        Value::SpecialForm(ref keyword) => {
            Err(format!("Invalid syntax with special form {}", keyword))
        }
        Value::ConsCell(ConsCell::Pair(ref car, ref cdr)) => {
            // we're going to be a special form or a function call; we need
            // to check that the arguments are a "true" list (i.e. ends with Nil)
            let cdr_cell = match *cdr.as_ref() {
                Value::ConsCell(ref cell) => cell,
                _ => { return Err(format!("Found a pair, expected a list: {:?}", cdr)); }
            };

            let args_vec = match into_vec(&cdr_cell) {
                Some(vec) => vec,
                _ => {
                    return Err(format!("Expected a list. Bad syntax for arguments: {:?}", cdr_cell));
                }
            };

            debug!("Args = {:?}", args_vec);

            let evald_car = try!(eval(car.as_ref(), env.clone()));

            match evald_car {
                Value::Procedure(ref arg_names, ref body, ref env) => {
                    return call_proc(env, &arg_names, &cdr_cell, body);
                }
                Value::SpecialForm(keyword) => {
                    return unimplemented();
                }
                _ => { return Err(format!("Expected a procedure: {:?}", evald_car)); }
            };
        }
    }
}

fn print(value: Value) -> () {
    log!("{:?}\n", value);
}

fn print_help() {
    log!(":load (or :l) to load an external file\n");
    log!(":exit to quit\n");
}

fn repl(env: &Rc<Environment>) {
    let repl_env = Rc::new(Environment::new(Some(env.clone())));
    loop {
        log!("λ ");

        let mut raw_input_line = String::new();
        
        match io::stdin().read_line(&mut raw_input_line) {
            Err(error) => { log!("error: {}", error) }
            Ok(_) => (),
        }
        
        let input_line: &str = raw_input_line.trim();

        match input_line {
            "" | ":exit" => { log!("\nGoodbye."); break; }
            ":help" => { print_help(); }
            _ => {
                match parse(input_line) {
                    Ok(exprs) => {
                        for e in exprs.into_iter() {
                            // debug_dump(&e);
                            let result = eval(&e, repl_env.clone());
                            match result {
                                Ok(v) => print(v),
                                Err(e) => log!("Error: {}\n", e),
                            }
                        }
                    }
                    Err(e) => { debug_dump(e); }
                }
            }
        }
    }
}

fn main() {
    log!("(R)ust (E)xperiments in (LISP)\n");
    debug!("DEBUG activated");
    
    let global_env = Rc::new(Environment::new(None));

    repl(&global_env);
}
