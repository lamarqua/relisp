#![allow(unused_variables)]
#![allow(dead_code)]

extern crate regex;

use regex::{Regex, RegexBuilder};
use std::collections::HashMap;
use std::fmt::Debug;
use std::io::prelude::*;
use std::io;
use std::rc::Rc;
use std::fs::File;
use std::cell::RefCell;


// -- UTILS -- 
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

// -- Data structures -- 
#[derive(Debug, Clone)]
enum Primitive {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
}

#[derive(Debug, Clone)]
enum ConsCell {
    Nil,
    Pair(Box<Value>, Box<Value>),
}

fn to_vec(list: &ConsCell) -> Option<Vec<Value>> {
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

struct NativeProc(fn(Vec<Value>, &SharedEnv) -> Result<Value, String>);

#[derive(Clone)]
enum ProcedureCode {
    InterpretedProc(Vec<String>, Vec<Value>, SharedEnv),
    NativeProc(NativeProc),
}

impl Debug for ProcedureCode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        match self {
            &ProcedureCode::InterpretedProc(ref arg_names, ref body_exprs, _) => {
                write!(f, "args: {:?}, body: {:?}", arg_names, body_exprs)
            }
            &ProcedureCode::NativeProc(ref np) => {
                write!(f, "{:?}", np)
            }
        }
    }
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
    Primitive(Primitive),
    ConsCell(ConsCell),
    Symbol(String),
    Quote(Box<Value>),
    Procedure(ProcedureCode),
    SpecialForm(String),
}   


// -- Lexing, parsing & interpreter --

#[derive(Clone, Debug, PartialEq)]
enum TokenType {
    Close,
    Comment,
    Dot,
    Eof,
    BoolLiteral,
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
            (r#"^true|false|\#true|\#false|\#t|\#f"#, TokenType::BoolLiteral),
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
    parent: Option<SharedEnv>
}

impl Environment {
    fn new(parent: Option<SharedEnv>) -> Self {
        Environment {
            table: HashMap::new(),
            parent: parent,
        }
    }

    fn insert(&mut self, symbol: &str, value: Value) {
        self.table.insert(symbol.to_string(), value);
    }

    fn lookup(&self, symbol: &str) -> Option<Value> {
        if self.table.contains_key(symbol) {
            return self.table.get(symbol).cloned()
        } else {
            if let Some(ref parent) = self.parent {
                return parent.borrow().lookup(symbol);
            }
        }
        return None;
    }
}

type SharedEnv = Rc<RefCell<Environment>>;

fn parse(input_string: &str) -> Result<Vec<Value>, String> {
    let mut lexer = Lexer::new(input_string);

    fn parens_match(o: &str, c: &str) -> bool {
        match (o, c) {
            ("(", ")") | ("[", "]") | ("{", "}") => true,
            _ => false
        }
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
        match token_type { // we also parse the primitive types directly
            TokenType::Int => {
                let v = Value::Primitive(Primitive::Int(matched.parse::<i64>().unwrap()));
                return Ok(v);
            }
            TokenType::Float => {
                let v = Value::Primitive(Primitive::Float(matched.parse::<f64>().unwrap()));
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
            TokenType::BoolLiteral => {
                let v = match matched {
                    "#t" | "#true" | "true" => Value::Primitive(Primitive::Bool(true)),
                    _ => Value::Primitive(Primitive::Bool(false)),
                };
                return Ok(v);
            }
            TokenType::String => {
                let v = Value::Primitive(Primitive::String(matched[1..matched.len()-1].to_string()));
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

fn eval(value: &Value, env: SharedEnv) -> Result<Value, String> {
    fn call_proc(args: &[Value], procedure: &ProcedureCode, call_env: &SharedEnv) -> Result<Value, String> {
        match procedure {
            &ProcedureCode::InterpretedProc(ref arg_names, ref exprs, ref capture_env) => {
                let local_env = Rc::new(RefCell::new(Environment::new(Some(capture_env.clone()))));

                {
                    let mut borrowed_local_env = local_env.borrow_mut();

                    for (arg_name, arg_expr) in arg_names.iter().zip(args.iter()) {
                        let arg_val = try!(eval(arg_expr, call_env.clone()));
                        borrowed_local_env.insert(arg_name, arg_val);
                    }
                }

                // debug_dump(&capture_env);
                let mut res: Result<Value, String> = Err(format!("Internal error. (Empty procedure body)"));
                for e in exprs.iter() {
                    res = eval(e, local_env.clone());
                    if res.is_err() {
                        return res;
                    }
                }
                return res;
            }
            &ProcedureCode::NativeProc(NativeProc(ref f)) => {
                let mut evald_args = Vec::new();
                for arg in args {
                    let v = try!(eval(&arg, call_env.clone()));
                    evald_args.push(v);
                }
                return f(evald_args, &call_env);
            }
        }
    }

    fn call_special_form(env: SharedEnv, keyword: &str, sf_args: &[Value]) -> Result<Value, String> {
        match keyword {
            "lambda" => {
                let mut arg_names: Vec<String> = Vec::new();

                match &sf_args[0] {
                    &Value::ConsCell(ref fun_args) => { 
                        let arg_names_values: Vec<Value> = try!(to_vec(fun_args)
                            .ok_or(format!("Got a non-pure list for arguments. {:?}", fun_args)));

                        for arg in arg_names_values {
                            match arg {
                                Value::Symbol(ref s) => arg_names.push(s.clone()),
                                _ => {
                                    return Err(format!("Non-symbol argument found in argument list. {:?}", arg_names))
                                }
                            }
                        }
                    },
                    _ => { return Err(format!("Expected list of arguments for lambda. {:?}", sf_args[0])) }
                };

                let proc_body = sf_args[1..].to_vec();
                debug_dump(&proc_body);

                let proc_code = ProcedureCode::InterpretedProc(arg_names, proc_body, env.clone());
                return Ok(Value::Procedure(proc_code));
            }

            "define" => {
                let symbol_name = match &sf_args[0] {
                    &Value::Symbol(ref s) => s,
                    _ => {
                        return Err(format!("Expected a symbol for define. {:?}", &sf_args[0]));
                    }
                };
                if sf_args.len() > 2 {
                    return Err(format!("Too many expressions for define. {:?}", sf_args));
                }

                let res = try!(eval(&sf_args[1], env.clone()));
                env.borrow_mut().insert(symbol_name, res.clone());

                return Ok(res);
            }

            "quote" => {
                if sf_args.len() > 1 {
                    return Err(format!("Too many arguments for quote {:?}", sf_args));
                }
                return Ok(Value::Quote(Box::new(sf_args[0].clone())));
            }

            "if" => {
                if sf_args.len() != 3 {
                    return Err(format!("Invalid number of arguments for if {:?}", sf_args));
                }
                if let Value::Primitive(Primitive::Bool(b)) = sf_args[0] {
                    if !b {
                        return eval(&sf_args[2], env.clone());
                    }
                }
                return eval(&sf_args[1], env.clone());
            }

            &_ => {
                debug_dump(format!("Tried to call unknown special form {} in call_special_form", keyword));
                return Err("Internal error".to_string());
            }
        }

    }

    match *value {
        Value::Primitive(_) => Ok(value.clone()),
        Value::Quote(ref quoted_val) => Ok(*quoted_val.clone()),
        Value::Procedure(..) => Ok(value.clone()),
        Value::ConsCell(ConsCell::Nil) => Ok(value.clone()),
        Value::Symbol(ref symbol_name) => env.borrow().lookup(&symbol_name)
                                        .ok_or(format!("Unknown symbol {}", &symbol_name))
                                        .map(|x| x.clone()),
        Value::SpecialForm(ref keyword) => {
            Err(format!("Invalid syntax with special form {}", keyword))
        }
        Value::ConsCell(ref cell) => {
            let call_vec = match to_vec(cell) {
                Some(vec) => vec,
                None => { 
                    return Err(format!("Found empty list. Bad syntax for function / special form call: {:?}", cell));
                }
            };

            if let Value::SpecialForm(ref keyword) = call_vec[0] {
                return call_special_form(env.clone(), keyword, &call_vec[1..]);                
            }
            else {
                let evald_first_elem = try!(eval(&call_vec[0], env.clone()));

                if let Value::Procedure(ref procedure) = evald_first_elem {
                    return call_proc(&call_vec[1..], procedure, &env);
                }
                else {
                    return Err(format!("Found non-callable object. Bad syntax for function / special form call: {:?}", evald_first_elem));                    
                }
            }
        }
    }
}

fn print_value(value: &Value) -> () {
    log!("{:?}\n", value);
}

fn print_help() {
    log!("\\n or :exit to quit\n");
}

fn repl(env: &SharedEnv) {
    // let repl_env = Rc::new(Environment::new(Some(env.clone())));
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
                            let result = eval(&e, env.clone());
                            match result {
                                Ok(v) => print_value(&v),
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

// -- RUNTIME --
fn sum(args: Vec<Value>, env: &SharedEnv) -> Result<Value, String> {
    if args.len() == 0 {
        return Err(format!("Missing arguments for + function"));
    }
    match args[0] {
        Value::Primitive(Primitive::Int(i)) => {
            let mut acc: i64 = i;
            for v in args[1..].iter() {
                match v {
                    &Value::Primitive(Primitive::Int(j)) => { acc += j; }
                    _ => { return Err(format!("Inconsistent types for summation {:?}", v)); }
                }
            }
            return Ok(Value::Primitive(Primitive::Int(acc)));
        }
        Value::Primitive(Primitive::Float(f)) => {
            let mut acc: f64 = f;
            for v in args[1..].iter() {
                match v {
                    &Value::Primitive(Primitive::Float(j)) => { acc += j; }
                    _ => { return Err(format!("Inconsistent types for summation {:?}", v)); }
                }
            }

            return Ok(Value::Primitive(Primitive::Float(acc)));
        }
        _ => {
            return Err(format!("Type is not summable {:?}", &args[0]));
        }
    }
}

fn init_env(env: &SharedEnv) -> () {
    let mut borrow = env.borrow_mut();
    borrow.insert("nil", Value::ConsCell(ConsCell::Nil));
    borrow.insert("+", Value::Procedure(ProcedureCode::NativeProc(NativeProc(sum))));
}

// -- REPL --
fn read_file(filename: &str) -> Result<String, String> {

    let mut file = try!(File::open(filename).map_err(|e| e.to_string()));

    let mut contents = String::new();
    let _ = try!(file.read_to_string(&mut contents).map_err(|e| e.to_string()));

    return Ok(contents);
}

fn eval_file(contents: &str, env: &SharedEnv) -> Result<(), String> {
    let exprs = try!(parse(&contents));

    for e in exprs.into_iter() {
        print_value(&try!(eval(&e, env.clone())));
    }

    return Ok(());
}

fn load_file(filename: &str, env: &SharedEnv) -> Result<(), String> {
    let contents = try!(read_file(filename));
    return eval_file(&contents, env);
}


fn main() {
    log!("(R)ust (E)xperiments in (LISP)\n");
    debug!("DEBUG activated");

    let global_env = Rc::new(RefCell::new(Environment::new(None)));

    init_env(&global_env);

    if let Err(e) = load_file("src/prelude.lisp", &global_env) {
        log!("Error while loading prelude file. {}\n", e);
    }

    repl(&global_env);
}
